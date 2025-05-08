from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import accelerate
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import re
import os
import pdb
import json
from tqdm import tqdm

# convert the text into an embedding vector either uses a model or pipeline
def extract_embeddings(strings, pipe=None, model=None):
    # extract embeddings using either pipe or model
    if pipe != None:
        output = pipe(strings)
    elif model != None:
        output = model.encode(strings)
    averaged = []
    # the output from the embedding extraction is an embedding vector for every token in the string
    # we want to extract the average token embedding to get a one dimensional vector for similarity comparison
    # there can also be multiple strings passed into the pipeline
    for entry in output:
        try:
            # average across all tokens
            averaged_entry = np.mean(entry[0], axis=0)
        except:
            # some cases some pipelines already average for you
            averaged_entry = entry
        # store the average embedding for the given string
        averaged.append(averaged_entry)
    return np.array(averaged)

# load and create model and tokenizer objects
def initialize_model_objects(name, quant_config, attention=True):
    # load tokenizer for the model to use
    tokenizer = AutoTokenizer.from_pretrained(name)
    # load model with quantization and device mapping
    # extract attentions
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant_config, device_map="auto", output_attentions=attention, torch_dtype=torch.float16) # load LLM on GPU
    return model, tokenizer

# function to standardize predictions
def predict(query, model, tokenizer, hps, system_prompt=None, context=None, examples=None, rag=False, pipe=None):
    # format query for the model accordingly
    text, tokens, context_numbers = format_query(query, tokenizer, context=context, system_prompt=system_prompt, examples=examples, rag=rag, p=pipe)
    # pass in tokens to generate an answer using the formatted prompt
    output = model.generate(**tokens, **hps, output_attentions=True, return_dict_in_generate=True)
    return output, context_numbers

def format_query(query, tokenizer, context=None, system_prompt=None, examples=None, rag=False, p=None):
    messages = []
    # system prompt formatting
    # an LLM prompt does not have to contain a system prompt so check for this
    if system_prompt != None:
        # system prompt can also have examples for a task
        if examples != None:
            # if there are examples, iterate
            for example in examples:
                # add example to system prompt
                system_prompt = system_prompt + "\n\n" + f"{example}"
    # store the system prompt
    messages.append({"role": "system",
            "content": system_prompt})
    # a propmt could have context to help answer a specific query
    if context != None:
        # the formatting can also incorporate retrieval augmented generation (RAG) which also requires a pipeline for embedding extraction
        if rag and p != None:
            # rag using clustering or similarity selection methods below
            context_bool = cluster_loop({"question": query, "all_evidence": context}, pipe=p)
            ##### context_bool = rag_before_loop(context, query)
            # the output from either selection method is a boolean vector
            context = list(np.array(context)[context_bool])
        # append all or selected context entries as a string depending on whether rag is performed
        context = "\n".join(context) + "--"
        # the orignal query and the associated context are appended
        query = query + "\n\n" + "Here are clinical notes you have written about this patient in the past:\n" + context + "\n"
    # append the query
    messages.extend([
            {"role": "user",
            "content": query}])
    # create a custom template in case there is not template for the model
    custom_template = """{% for message in messages %}
    {% if message['role'] == 'system' %}
    System: {{ message['content'] }}
    {% elif message['role'] == 'user' %}
    User: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
    Assistant: {{ message['content'] }}
    {% endif %}
    {% endfor %}
    """
    # tokenize input text and extract the string template as well
    try:
        input_text=tokenizer.apply_chat_template(messages, tokenize=False)
        inputs=tokenizer(input_text, return_tensors='pt').to("cuda")
    # if the model attempted does not contain a template
    except:
        tokenizer.chat_template = custom_template
        input_text=tokenizer.apply_chat_template(messages, tokenize=False)
        inputs=tokenizer(input_text, return_tensors='pt').to("cuda")
    # return input text, input tokens, and selected entry ids if rag is used
    if not rag:
        return input_text, inputs, None
    else:
        return input_text, inputs, np.where(context_bool == True)[0] + 1

# this function allows me to find token sequences that are the most similar; important for extract indices of context tokens
def check(l1, l2, threshold=0.5):
    equal = sum(np.equal(l1, l2))
    if equal/len(l1) > threshold:
        return True
    else:
        return False

# this function enables me to extract the token indices of the context and output
def extract_context_and_output_indices(input_tokens, output_tokens, tokenizer, context_delimiter = "--", output_delimiter = "\.", output_start = "<|start_header_id|>assistant<|end_header_id|>"):
    indices = []
    # iterate all the input token entries
    for i, t in enumerate(input_tokens):
        # detokenize into a string
        value = tokenizer.decode(t)
        # check for a special context entry delimiter that I kept consistent in data processing
        if context_delimiter in value:
            indices.append(i)
    # use the indices of the special token to find all the context entry tokens
    context_coordinates = np.array([[indices[element-1], indices[element]] for element in range(1, len(indices))])
    # next I encode the header that dictates the start of the output for the model
    start = tokenizer.encode(output_start)[1:]
    # find the starting point of the output
    shift = len(input_tokens) + len(start)
    # find all tokens in the output after the starting point
    assistant_tokens = output_tokens[shift:].tolist()
    # convert it into text
    assistant_text = tokenizer.decode(assistant_tokens)[1:]
    # split the output by sentence
    substrings = re.split(output_delimiter, assistant_text)
    output_coordinates = []
    # iterate output strings to find the token indices for the output
    for substring in substrings:
        # convert back to a token array
        substring_tokens = tokenizer.encode(substring)[1:]
        # iterate all the tokens in the output and try to find the indices of the current output sentence
        for i in range(len(assistant_tokens) - (len(substring_tokens)-1)):
            if check(substring_tokens, assistant_tokens[i:i+len(substring_tokens)]):
                output_coordinates.append([i+shift, i+shift+len(substring_tokens)])
    # need to make sure the tokens are at least greater than size 0
    if output_coordinates[-1][1] - output_coordinates[-1][0] == 1:
        output_coordinates = output_coordinates[:-1]
    return np.array(context_coordinates), np.array(output_coordinates)

# select layers based on how similar they are sequentially
def attention_dropout(attention_vectors, similarity_threshold):
    # extract all layer outputs
    layers = [attention_vectors[0]]
    # iterate all attention layers
    for i in range(0, len(attention_vectors)-1):
        # compare the current layer to the previous and check how much it changed
        score = cosine_similarity(attention_vectors[i], attention_vectors[i+1])
        # if the similarity is low that means that the attention output is unique and should be kept for averaging
        if score <= similarity_threshold:
            layers.append(attention_vectors[i+1])
    return layers

def attention_loop(input_tokens, output_tokens, attention_matrix, output_indices, context_indices, tokenizer, drop_attentions=True, threshold=1.64, context_numbers=None): # which attention layers, which question, 
    predictions = []
    layers_used = {}
    finalized_output = ""
    # iterate the token indicies for the output sentences
    for output_pair in output_indices:
        average_attention_per_context = []
        # iterate the token indices for the context sentences
        for i, context_pair in enumerate(context_indices):
            # parse the attention matrix to find the attention for each token in the output sentence
            output_tokens_of_interest = attention_matrix[output_pair[0] - len(input_tokens): output_pair[1] - len(input_tokens)]
            unwrapped = []
            # for each token attention matrix
            for token in output_tokens_of_interest:
                # extract the attention values that the current output token pays to the current context entry
                # average the attention values for the current output token across all the current context tokens for all attention heads in the layers
                # now I have average attention values for the current output token and all current context tokens for all attention layers and heads
                # the dimensions of the attention matrix are output token(1 because we are looking per token):attention layers:attention heads:input tokens
                layers = np.array([layer[:,:,:,context_pair[0]:context_pair[1]].mean(axis=1).tolist() for layer in token])
                # now I have 32 vectors (because 32 attention layers), with dimension 32 because averaged across all 32 attention heads and context tokens
                layers = layers.reshape(layers.shape[0], layers.shape[-1])
                unwrapped.append(layers)
            # average across all the current output sentence's tokens
            unwrapped = np.mean(unwrapped, axis=0)
            # now we have average attention layers for the entire output sentence and current context entry
            # I now run selection for the attention layers that are most different
            if drop_attentions:
                layers = np.array(attention_dropout(unwrapped, 0.9))
            # then I average all the attention layers selected and attention heads to get a final average attention score for the output sentence and context entry
            average_attention_per_context.append(np.array(layers).mean())
        # i convert the average attention scores for current output sentence and all context entries into z-scores
        portions = (np.array(average_attention_per_context) - np.mean(average_attention_per_context))/np.std(average_attention_per_context)
        # i select the context entries that have z-scores higher than threshold (this allows me to select context only if its needed for the sentence)
        above_threshold_indices = portions > threshold
        # in the case that rag or clustering was used prior to attention, we make sure we select the correct context entry numbers and not the index unless there was no selection
        try:
            if context_numbers == None:
                context_numbers = np.array(range(0, len(context_indices)))+1
        except:
            pass
        # now I add the citations selected to each output sentence
        # i check to make sure at least one evidence entry was above the threshold
        if np.max(portions) > threshold:
            # I append the evidence entry numbers that are most important for the sentence in output
            output_string = tokenizer.decode(output_tokens[output_pair[0]:output_pair[1]]).replace('\n', '')
            finalized_output = finalized_output + output_string + ". " + f"|{','.join([str(num) for num in list(context_numbers[above_threshold_indices])])}|" + "\n"
        # otherwise return the output sentence without citations
        else:
            output_string = tokenizer.decode(output_tokens[output_pair[0]:output_pair[1]]).replace('\n', '')
            finalized_output = finalized_output + output_string + ". | |\n"
        predictions.extend(context_numbers[above_threshold_indices])
    # output the string with citations and the arrow of improtant citations
    predictions = sorted(set(predictions))
    return list([str(p) for p in predictions]), finalized_output

def cluster_loop(data, pipe=None, model=None):
    # extract question and evidence
    question = data["question"]
    context_documents = data["all_evidence"]
    # combine the qeustion and context
    cluster_sequences = [question, *context_documents]
    # extract the embeddings for all of them
    cluster_sequence_embeddings = extract_embeddings(cluster_sequences, pipe=pipe, model=model)
    # convert the embeddings into principal components
    cluster_sequence_embeddings = PCA().fit_transform(cluster_sequence_embeddings)
    # cluster the question and context vectors
    fit = AgglomerativeClustering(n_clusters=2, metric="manhattan", linkage="single").fit(cluster_sequence_embeddings)
    # find the cluster with the question vector and all other members of the cluster are considered the important citations
    positive_indices = np.where(fit.labels_ == fit.labels_[0])[0]
    # store the cited evidence
    predictions = []
    for i in range(len(data["all_evidence"])):
        if i+1 in positive_indices:
            predictions.append(True)
        else:
            predictions.append(False)
    if sum(predictions) == 0:
        predictions = np.array([True]*len(data["all_evidence"]))
    return np.array(predictions)

# cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, np.transpose(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

# bert score function
def BERTscore(string1, string2):
    scorer = BERTScorer(model_type='roberta-large')
    P, R, F1 = scorer.score([string1], [string2])
    return F1.mean()

# Retrieval Augmented Generation
def rag_before_loop(context_entries, input_query, threshold=0):
    # iterate all context entries
    context_similarities = []
    for context_entry in context_entries:
        # calculate bert score for each context entry compared to the selected input
        context_similarities.append(BERTscore(context_entry, input_query))
    # same z-score function
    portions = (np.array(context_similarities) - np.mean(context_similarities))/np.std(context_similarities)
    if sum(np.array(portions) > threshold) == 0:
        return np.array([True]*len(portions))
    else:
        return np.array(portions) > threshold

def mutli_step(system_prompt, query, context, model, tokenizer, pipe, hps, output_end = "<|eot_id|>", output_start = "<|start_header_id|>assistant<|end_header_id|>", examples=None):
    # extract context entry numbers
    context_indices = np.array(list(range(len(context))))+1
    # find the selected context entries using clustering
    bool_list = cluster_loop({"question": query, "all_evidence": context}, pipe=pipe)
    # select the context entries
    selected_context = np.array(context)[bool_list]
    # find the indices of the selected context entries
    selected_context_indices = [str(val) for val in context_indices[bool_list]]
    # if none selected then just use all context entries (happens rarely)
    if len(selected_context_indices) <= 0:
        print(selected_context_indices)
        selected_context = context
        selected_context_indices = [str(i) for i in context_indices]
    # combind selected context into a string
    string_context = "\n".join(selected_context)
    # make prediction using the selected context
    output, _ = predict(f"Here is the question to answer:\n{query}\n\nHere are the clinical notes:\n{string_context}", model, tokenizer, hps, system_prompt=system_prompt, examples=examples)
    # decode the output
    string = tokenizer.decode(output["sequences"][0])
    # clean up the output
    string = string.replace("\n", "")
    li = string.split(".")
    # add all selected context to the end
    string = ". | |\n".join(li)
    string = string[string.index(output_start)+len(output_start):string.rindex(output_end)] + " |" + ",".join(selected_context_indices) + "|"
    return string


if __name__ == "__main__":

    # loading models and hyperparameters
    c = BitsAndBytesConfig(load_in_4bit=False)
    hps = dict(max_new_tokens=1000, temperature=0.001, top_k=10, num_return_sequences = 1, do_sample=True)
    model, tokenizer = initialize_model_objects("meta-llama/Meta-Llama-3.1-8B-Instruct", c)
    pipe = pipeline("feature-extraction", model="dmis-lab/biobert-v1.1", device_map="auto")
    
    # data
    data = json.load(open("data/dev/dev_reformatted.json"))

    # all prompts and examples for the approaches
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
    example_question = "Why did they perform the emergency salvage repair on him?\nTook my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home."
    example_context = "1: He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.\n2: He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.\n3: Please see operative note for details which included cardiac arrest x2.\n4: Postoperatively he was taken to the intensive care unit for monitoring with an open chest.\n5: He remained intubated and sedated on pressors and inotropes.\n6: On 2025-1-22, he returned to the operating room where he underwent exploration and chest closure.\n7: On 1-25 he returned to the OR for abd closure JP/ drain placement/ feeding jejunostomy placed at that time for nutritional support.\n8: Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema.\n9: Packed with dry gauze and covered w/DSD"
    with_context_answer = "1: He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. |1|\n2: He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. |2|\n8: Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema. |8|"
    without_context_answer = "1: He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.\n2: He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.\n8: Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema."
    system_prompt = "You are a doctor who is answering a patient question. Be detailed and quote text from the context you think are useful to answer the question since you are evaluated on how well your answer aligns with the clinical notes. Emphasize clarity, conciseness, and semantic matching with the source material. This is very important: your answer must be less than 75 words.\n"
    cluster_prompt = "Summarize the clinical notes in less than 75 words. Please preserve details from the clinical notes as it will be used to evaluate your response. You are recommended to quote segments. Below is an example of an ideal summarization."

    example_prompting = f"""
Here is an Example Clinician Question
{example_question}
Here is an Example Clinical Note |sentences numbered for grounding|
{example_context}
Here is how your answers should be formated Answer |numbered citation to highlight relevant sentences from clinical note|
{with_context_answer}
"""

    example_summarizing = f"""
Here is an Example Clinical Note:
{example_context}
Here is how your answers should be formated:
{without_context_answer}"""

    example_selecting = f"""
Here is an Example Clinician Question:
{example_question}
Here is an Example Clinical Note:
{example_context}
Here is how your answers should be formated:
{without_context_answer}"""   
    
    all_outputs = []
    for number, entry in tqdm(enumerate(data), total=len(data)):
        # question combining both
        question = entry["clinician_question"] + "\n" + entry["patient_question"]
        # all context lists
        context_text = entry["evidence_blurb"].split("\n")
        # query with explanation
        query = f"Answer the question in the following message:\n{question}"
        # output storage
        outputs = {"Attention": None, "Cluster": None, "Multi": None}

        # cluster-only
        cluster_out = mutli_step(cluster_prompt, question, context_text, model, tokenizer, pipe, hps, examples=None)
        print(cluster_out)
        outputs["Cluster"] = {"case_id": entry["case_id"], "answer": f"{cluster_out}"}
        print("----------------------------------")

        # attention-only
        output_matrix, context_numbers = predict(query, model, tokenizer, hps, system_prompt=system_prompt, context=context_text, examples=[example_selecting], rag=False, pipe=None)
        input_text, input_tokens, _ = format_query(query, tokenizer, context=context_text, system_prompt=system_prompt, examples=[example_selecting], rag=False, p=None)
        context, output = extract_context_and_output_indices(input_tokens["input_ids"][0], output_matrix["sequences"][0], tokenizer, output_start=assistant_header)
        attention_predictions, attention_output = attention_loop(input_tokens["input_ids"][0], output_matrix["sequences"][0], output_matrix.attentions, output, context, tokenizer, drop_attentions=True, threshold=0, context_numbers=context_numbers)
        print(f"--Output--:\n{attention_output}")
        outputs["Attention"] = {"case_id": entry["case_id"], "answer": attention_output}
        print("------------------------------------------------------")

        # cluster + attention
        output_matrix, context_numbers = predict(query, model, tokenizer, hps, system_prompt=system_prompt, context=context_text, examples=[example_selecting], rag=True, pipe=pipe)
        input_text, input_tokens, _ = format_query(query, tokenizer, context=context_text, system_prompt=system_prompt, examples=[example_selecting], rag=True, p=pipe)
        context, output = extract_context_and_output_indices(input_tokens["input_ids"][0], output_matrix["sequences"][0], tokenizer, output_start=assistant_header)
        attention_predictions, attention_output = attention_loop(input_tokens["input_ids"][0], output_matrix["sequences"][0], output_matrix.attentions, output, context, tokenizer, drop_attentions=True, threshold=0, context_numbers=context_numbers)
        print(f"--Output--:\n{attention_output}")
        outputs["Multi"] = {"case_id": entry["case_id"], "answer": attention_output}
        print("------------------------------------------------------")
        all_outputs.append(outputs)
    
    # return
    with open("dev_multi.json", "w") as file:
        json.dump([out["Multi"] for out in all_outputs], file)
    with open("dev_attention.json", "w") as file:
        json.dump([out["Attention"] for out in all_outputs], file)
    with open("dev_cluster.json", "w") as file:
        json.dump([out["Cluster"] for out in all_outputs], file)