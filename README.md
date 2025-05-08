# loyola_archehr_2025
Unsupervised Attribution of LLM Generated Text
# How to Run
1. Virtual Environment
If there are any missing packages during runtime, please install using pip.
```
python -m venv [name of environment]
```
```
pip install -r requirements.txt
```

2. Data Processing
First edit input and output paths in the xml to json python script.
```
python scripts/xml_to_json.py
```

Then edit the process data script to add in the new json file, the new path for the reformatted json file, and the label path. If there are no labels just comment out the code that contains label parsing.
```
python scripts/process_data.py
```

3. Running Experiments from Paper
You can edit the hyperparameters throughout the code if desired. The only thing that would need to be changed is the path to the input data which is on line 312.
```
python scripts/run_experiments.py
```
Some hyperparameters that can be edited include:
- system prompt
- threshold in attention loop
- drop attentions boolean (to select for attention layers)
- providing an example in prompt (replace list with None, otherwise put examples in a list)
- cluster prior to generation (encoded as rag boolean) and pipeline parameters
- models loaded (may need to change the assistant header if doing this)
- generation hyperparamters encoded as the hps variable
- clustering algorithm and its hyperpameters can be adjusted in the cluster loop function
