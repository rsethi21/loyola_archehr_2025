import xmltodict
import json
from tqdm import tqdm

# convert xml file to json for ease of parsing
def convert(in_path, out_path):
    with open(in_path) as in_file:
        text = xmltodict.parse(in_file.read())
    as_json = json.dumps(text, indent=4)
    open(out_path, "w").write(as_json)

# repetitive function to clear out any newline characters
def reformat_string(string):
    return string.replace("\n", " ")

# reformat json to have standardized parsing
def reformat_data(in_path, out_path, label_path):
    # load cases and label file
    cases = json.load(open(in_path))["annotations"]["case"]
    labels = json.load(open(label_path))
    reformated_data = []
    # iterate the case and associated label
    for case, label in tqdm(zip(cases, labels), total=len(cases)):
        # shell dictionary to store reformatted data
        entry = {}
        # store case id
        entry["case_id"] = case["@id"]
        # store patient question; patient questions were either a dictionary with one question or multiple questions
        if type(case["patient_question"]["phrase"]) == dict:
            entry["patient_question"] = case["patient_question"]["phrase"]["#text"] # could be a hyperparameter to tune, along with attention threshold for inclusion
        else:
            entry["patient_question"] = " ".join([q["#text"] for q in case["patient_question"]["phrase"]])
        # store clinician question
        entry["clinician_question"] = case["clinician_question"]
        # extract evidence entries from the case and store as a list; clean up and remove newline characters
        entry["all_evidence"] = [sentence_entry['#text'].replace('\n', ' ') for sentence_entry in case["note_excerpt_sentences"]["sentence"]]
        # store all evidence as a string separated by newline characters and add id of the sentence for citation purposes
        entry["evidence_blurb"] = "\n".join([f"-- {int(sentence_entry['@id'])}: {reformat_string(sentence_entry['#text'])}" for sentence_entry in case["note_excerpt_sentences"]["sentence"]])
        # store the importance scores for the evidence entries
        entry["importance"] = label["answers"]
        # list of evidence entries that are both supplementary and essential
        entry["answers_lenient"] = [int(sentence["sentence_id"]) for sentence in label["answers"] if sentence["relevance"] != "not-relevant"]
        # list of evidence entries that are essential
        entry["answers_strict"] = [int(sentence["sentence_id"]) for sentence in label["answers"] if sentence["relevance"] == "essential"]
        # append the reformated case entry
        reformated_data.append(entry)
    # save the reformated data as a json
    json.dump(reformated_data, open(out_path, "w"))

if __name__ == "__main__":
    reformat_data("data/dev/dev_as_json.json", "data/dev/dev_reformatted.json", "data/dev/archehr-qa_key.json")