import json
import os
from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed



def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None
    

def decompose_vector(df):
    components = df['vector'].str.extract(
        r'AV:(?P<AV>[A-Z])\/AC:(?P<AC>[A-Z])\/PR:(?P<PR>[A-Z])\/UI:(?P<UI>[A-Z])\/S:(?P<S>[A-Z])\/C:(?P<C>[A-Z])\/I:(?P<I>[A-Z])\/A:(?P<A>[A-Z])'
    )
    return pd.concat([df, components], axis=1)


def extract_links(obj):
    links = []
    if isinstance(obj, dict):
        for value in obj.values():
            links.extend(extract_links(value))
    elif isinstance(obj, list):
        for item in obj:
            links.extend(extract_links(item))
    elif isinstance(obj, str) and (obj.startswith('http://') or obj.startswith('https://')):
        links.append(obj)
    return links


def fetch_and_parse_cwe(row):
    cwe_id = row['cwe_id']
    if not cwe_id or not cwe_id.startswith('CWE-'):
        return {
            'cwe_description': None,
            'cwe_enhanced_description': None,
            'cwe_detection': None,
            'cwe_consequences': None,
            'cwe_mitigations': None
        }

    try:
        cwe_num = int(cwe_id.split('-')[1])
        url = f"https://cwe.mitre.org/data/definitions/{cwe_num}.html"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        text_content = soup.get_text()
        return extract_cwe__sections(text_content)
    except Exception as e:
        return {
            'cwe_description': None,
            'cwe_enhanced_description': None,
            'cwe_detection': None,
            'cwe_consequences': None,
            'cwe_mitigations': None
        }



def extract_cwe__sections(text):
    result = {
        'cwe_description': 'N/A',
        'cwe_enhanced_description': 'N/A',
        'cwe_consequences': 'N/A',
        'cwe_mitigations': 'N/A'
    }
    
    desc_match = re.search(r'Description\s*\n+(.*?)\n\n[A-Z]', text, re.DOTALL)
    enhanced_desc_match = re.search(r'Extended Description\s*\n+(.*?)\n\n[A-Z]', text, re.DOTALL)
    
    if desc_match:
        result['cwe_description'] = desc_match.group(1).strip()
    if enhanced_desc_match:
        result['cwe_enhanced_description'] = enhanced_desc_match.group(1).strip()
    
    try:
        consequences = text.split('achieve a different impact.')[1].split('Potential Mitigations')[0].strip().replace('\n', ' ')
        mitigations = text.split('Potential Mitigations')[1].split('Relationships')[0].strip().replace('\n', ' ')
        result['cwe_consequences'] = consequences
        result['cwe_mitigations'] = mitigations
    except IndexError:
        pass
    
    return result



# Save path for the processed dataset
output_path = './dataset.parquet'

# Directory to search for JSON files
directory = './cves/cves/'

json_file_paths = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.json') and file.startswith('CVE'):
            json_file_paths.append(os.path.join(root, file))

json_data = []
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(load_json, path) for path in json_file_paths]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading JSONs"):
        result = future.result()
        if result:
            json_data.append(result)

# Extract the data from the jsons
data = []
for record in tqdm(json_data, desc="Processing JSONs"):
    # Description
    cve_id = record['cveMetadata']['cveId']
    cve_descriptions = record['containers']['cna'].get('descriptions', [])
    cve_description = next((desc['value'] for desc in cve_descriptions if desc.get('lang') == 'en'), None)
    # CVSS (3.1) vector
    metrics = record['containers']['cna'].get('metrics', [])
    if not metrics:
        continue
    for metric in metrics:
        if 'cvssV3_1' in metric:
            cvss = metric['cvssV3_1']['baseScore']
            vector = metric['cvssV3_1']['vectorString']
        else:
            cvss = None
            vector = None
    # Assigner and CWE
    try:
        assigner = record['cveMetadata']['assignerShortName']
    except KeyError:
        continue
    try:
        cwe_id = record['containers']['cna']['problemTypes'][0]['descriptions'][0].get('cweId', '') if record['containers']['cna']['problemTypes'] else ''
    except KeyError:
        continue
    # Extracting external links
    links = extract_links(record)
    # Saving data
    data.append({'cve_id': cve_id,
                 'cve_description': cve_description,
                 'cwe_id': cwe_id,
                 'assigner': assigner,
                 'cvss': cvss,
                 'vector': vector,
                 'links': links,
                })
    
# Creating DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by='cve_id')
# Dataset cleaning
df = df[df['cwe_id'].notna() & df['cwe_id'].str.strip().astype(bool)]
df = df[df['assigner'].notna() & df['assigner'].str.strip().astype(bool)]
df = df[df['cvss'].notna()]

# Decomposing CVSS vector
cvss_components = ['AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A']
df = decompose_vector(df)

# Extracting CWE data
unique_cwes_df = df[['cwe_id']].drop_duplicates()

# Multithreaded extraction
cwe_results = {}
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(fetch_and_parse_cwe, row): row['cwe_id'] for _, row in unique_cwes_df.iterrows()}
    for future in tqdm(futures, desc="Processing Unique CWEs", total=len(futures)):
        cwe_id = futures[future]
        cwe_results[cwe_id] = future.result()

# Merge results back into the main df
print('Merging CWE data...')
df = df.join(df['cwe_id'].map(cwe_results).apply(pd.Series))

# Saving
print('Saving dataset...')
df.to_parquet(output_path)
print('Dataset saved to', output_path)