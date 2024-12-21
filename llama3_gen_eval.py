import yaml
import json
import random
import copy
import os
import subprocess
from generate import main as generate_main
from omegaconf import OmegaConf
import re
import pandas as pd
import time
SEED = 42
#first we set up a base config
def base_config(epoch):
    config = {
        "model": {
            "_component_": "torchtune.models.llama3.llama3_8b"
        },
        "checkpointer": {
            "_component_": "torchtune.training.FullModelMetaCheckpointer",
            "checkpoint_dir": "/scratch/qt2094/DLSYS/models/meta-llama/finetuned_models/",
            "checkpoint_files": ["meta_model_3.pt"],
            "adapter_checkpoint": f"/scratch/qt2094/DLSYS/models/meta-llama/finetuned_models/adapter_{epoch-1}.pt",
            "recipe_checkpoint": "/scratch/qt2094/DLSYS/models/meta-llama/finetuned_models/recipe_state.pt",
            "output_dir": "/scratch/qt2094/DLSYS/models/meta-llama/llama_finetune_logs",
            "model_type": "LLAMA3"
        },
        "device": "cuda",
        "dtype": "bf16",
        "enable_kv_cache": True,
        "seed": 1234,
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "/scratch/qt2094/DLSYS/models/meta-llama/original/tokenizer.model"
        },
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_k": 300,
        "quantizer": None
    }

    return OmegaConf.create(config)

def load_test_data(json_path, num_samples):
    random.seed(SEED)
    with open(json_path, 'r') as f:
        data = json.load(f)
    sample_idces = random.sample(range(len(data)), num_samples)
    data = [data[i] for i in sample_idces]
    return data, sample_idces

def split_chinese_sentences(text):
    sentences = re.split('([。！？；])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''] * (len(sentences[0::2]) - len(sentences[1::2])))]
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

#
def get_result_new(config, data_path, num_samples):
    data, sample_idces = load_test_data(data_path, num_samples)
    results_df = pd.DataFrame(columns=['sample_idx', 'ref_output', 'pred_output'])
    
    try:
        for i,sample in zip(sample_idces, data):
            # Create a deep copy of config
            sample_config = copy.deepcopy(config)
            
            # Add the prompt structure
            sample_config['prompt'] = {
                'system': "You are a medical AI assistant, your job is to Assist the doctor with their working process.",
                'user': sample['input']
            }
            
            # Convert to DictConfig
            cfg = OmegaConf.create(sample_config)
            
            try:
                # Pass DictConfig directly to generate_main
                result = generate_main(cfg)
                #cleaning the result
                result = re.sub(sample_config['prompt']['system'], '', result)
                result = re.sub(sample_config['prompt']['user'], '', result)
                #put result in the dataframe
                new_row = pd.DataFrame({
                    'sample_idx': [i],
                    'ref_output': [sample['output']],
                    'pred_output': [result]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
            except Exception as e:
                print(f"Error: {e}")
                new_row = pd.DataFrame({
                    'sample_idx': [i],
                    'ref_output': [sample['output']],
                    'pred_output': [f"Error: {e}"]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
    except Exception as e:
        print(f"An error occurred: {e}")

    results_list = results_df.to_dict('records')
    return results_list

def get_custom_result(prompt: str = None, epoch: int = 5) -> str:
    
    #prompt = """问：患者的情况概括在<病例描述>，请你根据根据患者情况回答问题。<病例描述>：（1）病例摘要：  患者，女，90岁，患者家属代诉患者于10天前无明显诱因下突发出现左侧肢体乏力，表现为左下肢难以上抬、不能行走，左上肢持物不稳，言语障碍无法说话，理解能力降低，无口角流延，无饮水呛咳，无头痛头晕，无胸闷胸痛，无视物旋转，无肢体抽搐，无大小便失禁，病后家属送至南宁市福利中医医院住院治疗，行头颅CT提示“两侧基底节区腔隙灶”，予以输液对症治疗后出院（具体诊疗经过不详），出院后遗留左侧肢体乏力，不能行走，瘫痪在床，生活无法自理，偶有单声发音，所说话语无法被听懂，无法正常交流，可自行进食，但进食较少，反应迟钝，现为求进一步治疗，遂送至我院就诊，门诊拟“脑梗死”收入我科。此次发病以来患者神清，精神差，食欲不振，睡眠可，大小便正常，近期体重未见明显增减。  （2）既往史  既往20余年前发现血压升高，最高血压达185/90mlg，在我院诊断为高血压病3级，曾服用硝苯地干缓释片降压，后自行停药多时，现未监测血压变化，自诉血压控制不详；有2型糖尿病病史8年，曾使用口服降糖药及胰岛素降糖，现已自行停药多时，血糖控制情况不详；有脑动脉供血不足、腰椎间盘突出症伴神经根病、老年性骨质疏松症、胸8椎体血管瘤、腰椎失稳、双下肢动脉粥样硬化、胆囊结石、脂肪肝、甲状腺左叶占位（性质待查）、右肺下叶慢性炎症等病史，否认肝炎、结核等传染病史，否认冠心病、肾炎等慢性病史，否认重大手术史、输血史、中毒史、外伤史，否认药物过敏史，预防接种史不详。（3）主诉   左侧肢体乏力、言语障碍10天 。（4）体格检查  体温36.0°C，脉搏70次/分，呼吸20次/分，血压154/82mmHg，神清，精神差，全身皮肤巩膜无黄染，浅表淋巴结未触及肿大，颈静脉无充盈，肝颈静脉征阴性，两肺呼吸音稍粗，双下肺可闻及少许湿性啰音，无干啰音，心界向左下扩大，心率70次/分，律齐，各瓣膜区未闻及病理性杂音，腹软，按压腹部无痛苦表情，肝脾肋下未触及肿大，墨菲氏征阴性，肝肾区无叩痛，移动性独音阴性，肠鸣音正常，双下肢无明显凹陷性水肿。神经系统：神清，精神差，呼之有反应，但言语含糊不清，无法交流，理解力降低，双侧瞳孔等大等圆，对光反射存在，双侧额纹、鼻唇沟对称，伸舌居中，左侧肢体肌张力偏高，左上肢4-级，左下肢3级，右侧肢体肌张力正常，肌力4级，生理反射存在，病理征未引出，脑膜刺激征阴性。舌质暗淡，苔薄白，脉細涩。   （5）辅助检查  头颅+腰椎CT（2024-11-01南宁市中医药）：1.两侧基底节区腔隙灶；2.脑萎缩；脑白质病变；建议MRI+DWI检查进一步评估，不除外急性-亚急性期脑梗死可能；3.左侧椎-基动脉及两侧颈内动脉虹吸部粥样硬化，建议CTA检查；4.腰1椎体压縮性骨折并积气；腰3/4-腰5骶1各椎间盘膨出。入院心电图：窦性心律，PtFV1增大，异常Q波。入院随机血糖：16.7mo1/L。  AI助手，根据病人的现病史和体格检查结果，请帮我进行初步诊断和依据 答： """
    config = base_config(epoch)
    sample_config = copy.deepcopy(config)
            
    # Add the prompt structure
    sample_config['prompt'] = {
        'system': "You are a medical AI assistant, your job is to Assist the doctor with their working process.",
        'user': prompt
    }
    
    # Convert to DictConfig
    cfg = OmegaConf.create(sample_config)
    result = generate_main(sample_config)
    result = re.sub(sample_config['prompt']['system'], '', result)
    result = re.sub(sample_config['prompt']['user'], '', result)
    print("Predicted result: ", result)

if __name__ == "__main__":
    prompt = """问：患者的情况概括在<病例描述>，请你根据根据患者情况回答问题。\n
    <病例描述>：\n
    （1）病例摘要：\n
    患者，女，90岁，患者家属代诉患者于10天前无明显诱因下突发出现左侧肢体乏力，表现为左下肢难以上抬、不能行走，左上肢持物不稳，言语障碍无法说话，理解能力降低，无口角流延，无饮水呛咳，无头痛头晕，无胸闷胸痛，无视物旋转，无肢体抽搐，无大小便失禁，病后家属送至南宁市福利中医医院住院治疗，行头颅CT提示"两侧基底节区腔隙灶"，予以输液对症治疗后出院（具体诊疗经过不详），出院后遗留左侧肢体乏力，不能行走，瘫痪在床，生活无法自理，偶有单声发音，所说话语无法被听懂，无法正常交流，可自行进食，但进食较少，反应迟钝，现为求进一步治疗，遂送至我院就诊，门诊拟"脑梗死"收入我科。此次发病以来患者神清，精神差，食欲不振，睡眠可，大小便正常，近期体重未见明显增减。\n\n
    （2）既往史：\n
    既往20余年前发现血压升高，最高血压达185/90mlg，在我院诊断为高血压病3级，曾服用硝苯地干缓释片降压，后自行停药多时，现未监测血压变化，自诉血压控制不详；有2型糖尿病病史8年，曾使用口服降糖药及胰岛素降糖，现已自行停药多时，血糖控制情况不详；有脑动脉供血不足、腰椎间盘突出症伴神经根病、老年性骨质疏松症、胸8椎体血管瘤、腰椎失稳、双下肢动脉粥样硬化、胆囊结石、脂肪肝、甲状腺左叶占位（性质待查）、右肺下叶慢性炎症等病史，否认肝炎、结核等传染病史，否认冠心病、肾炎等慢性病史，否认重大手术史、输血史、中毒史、外伤史，否认药物过敏史，预防接种史不详。\n\n
    （3）主诉：\n
    左侧肢体乏力、言语障碍10天。\n\n
    （4）体格检查：\n
    体温36.0°C，脉搏70次/分，呼吸20次/分，血压154/82mmHg，神清，精神差，全身皮肤巩膜无黄染，浅表淋巴结未触及肿大，颈静脉无充盈，肝颈静脉征阴性，两肺呼吸音稍粗，双下肺可闻及少许湿性啰音，无干啰音，心界向左下扩大，心率70次/分，律齐，各瓣膜区未闻及病理性杂音，腹软，按压腹部无痛苦表情，肝脾肋下未触及肿大，墨菲氏征阴性，肝肾区无叩痛，移动性独音阴性，肠鸣音正常，双下肢无明显凹陷性水肿。\n
    神经系统：神清，精神差，呼之有反应，但言语含糊不清，无法交流，理解力降低，双侧瞳孔等大等圆，对光反射存在，双侧额纹、鼻唇沟对称，伸舌居中，左侧肢体肌张力偏高，左上肢4-级，左下肢3级，右侧肢体肌张力正常，肌力4级，生理反射存在，病理征未引出，脑膜刺激征阴性。舌质暗淡，苔薄白，脉細涩。\n\n
    （5）辅助检查：\n
    头颅+腰椎CT（2024-11-01南宁市中医药）：\n
    1. 两侧基底节区腔隙灶；\n
    2. 脑萎缩；脑白质病变；建议MRI+DWI检查进一步评估，不除外急性-亚急性期脑梗死可能；\n
    3. 左侧椎-基动脉及两侧颈内动脉虹吸部粥样硬化，建议CTA检查；\n
    4. 腰1椎体压縮性骨折并积气；腰3/4-腰5骶1各椎间盘膨出。\n
    入院心电图：窦性心律，PtFV1增大，异常Q波。\n
    入院随机血糖：16.7mo1/L。\n\n
    AI助手，根据病人的现病史和体格检查结果，请帮我进行初步诊断和依据\n
    答："""
    start_time = time.time()
    prompt = "如果一个病人得了冠心病，他应该做哪些检查"
    get_custom_result(prompt, 5)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    exit()

    """json_path = '/scratch/qt2094/DLSYS/DLSys_Final/data/DoctorFlan_test.json'
    
    #the i th of epoch that you want to evaluate
    epoch = 5
    config = base_config(epoch)
    #result = generate_main(config)
    results = get_result_new(config, json_path, 1)
    
    output_path = f'/scratch/qt2094/DLSYS/DLSys_Final/eval_output/llama3_epoch{epoch}_eval.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    #print('output result: ', json_result)"""
