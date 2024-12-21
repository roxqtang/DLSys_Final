import pandas as pd
import json
import numpy as np
import datasets

# Read the parquet file
#df = pd.read_parquet("hf://datasets/FreedomIntelligence/DotaBench/data/test-00000-of-00001-aaa7e18cf6644d00.parquet")import pandas as pd

#splits = {'train': 'data/train-00000-of-00001-42015585e6c69743.parquet', 'test': 'data/test-00000-of-00001-31fd535e4fce7aa2.parquet'}
#DoctorFlan_train = pd.read_parquet("hf://datasets/FreedomIntelligence/DoctorFLAN/" + splits["train"])
#DoctorFlan_test = pd.read_parquet("hf://datasets/FreedomIntelligence/DoctorFLAN/" + splits["test"])
#CMtMedQA_train = pd.read_json("hf://datasets/Suprit/CMtMedQA/CMtMedQA.json")

# part 1
#knowledge_graph_dataset = datasets.load_dataset('FreedomIntelligence/huatuo_knowledge_graph_qa')
# part 2
#encyclopedia_dataset = datasets.load_dataset('FreedomIntelligence/huatuo_encyclopedia_qa')
# part 3 (only url)
#consultation_dataset = datasets.load_dataset('FreedomIntelligence/huatuo_consultation_qa')

DotaBench_test = pd.read_parquet("hf://datasets/FreedomIntelligence/DotaBench/data/test-00000-of-00001-aaa7e18cf6644d00.parquet")

df = DotaBench_test
# Sample 10% of the data randomly
#sample_size = int(len(df) * 0.01)
#df_sampled = df.sample(n=sample_size, random_state=42)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Convert sampled DataFrame to JSON format
json_data = df.to_dict(orient='records')
doctorflan_data_path = '/scratch/qt2094/DLSYS/DLSys_Final/data/DotaBench_test.json'
# Save to JSON file with custom encoder
with open(doctorflan_data_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)