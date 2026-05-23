import json

notebook_path = r'd:\Codes\Plant-Disease-Detection\Plant_Disease_Detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'google.colab' in line:
                new_source.append(f"# {line}")
            elif 'files.upload()' in line:
                new_source.append("    # " + line.lstrip())
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook colab imports commented out.")
