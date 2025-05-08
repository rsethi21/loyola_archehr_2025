import xmltodict
import json

input_path = "data/dev/archehr-qa.xml"
output_path = "data/dev/dev_as_json.json"

# read the xml file and parse
with open(input_path) as inp:
    df = xmltodict.parse(inp.read())

# save as a json
df_json = json.dumps(df, indent=4)
with open(output_path, "w") as out:
    out.write(df_json)