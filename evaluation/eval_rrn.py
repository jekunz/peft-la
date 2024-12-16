# -*- coding: utf-8 -*-
import tqdm
import torch
import transformers
import evaluate
from evaluate import load
import adapters
from bert_score import BERTScorer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    adapter = True
    my_adapter = "jekunz/lora256ff-igc"
    ds = load_dataset("thors/RRN")
    invert = False
    eval_all = False

    for shots in [0,1,5]:
        print('setup: ', shots)
        ds_examples= [] 
        for i in range(shots):
            ds_examples.append(ds['train'][i]) 

        ds_predict = []
        if eval_all: 
            for i in range(5,len(ds['train'])):
                ds_predict.append(ds['train'][i])
        else: 
            for i in range(5,505):
                ds_predict.append(ds['train'][i])

        if invert: 
            ds_examples = ds_examples[::-1]

        prompt_template = "Texta: {main}\nSamantekt: {intro}"
        if shots == 0: 
            prompt_examples = "Draga eftirfarandi texta saman í eina málsgrein:\n"
        else:
            prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
        
        if adapter: 
            adapters.init(model)
            adapter_name = model.load_adapter(my_adapter)
            model.set_active_adapters(adapter_name)
            print("Active Adapter:", model.active_adapters)
            model = model.to('cuda')

        pipeline  = transformers.pipeline("text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16)
        
        gen_config = {
        "temperature": 0.7,
        "top_p": 0.1,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "do_sample": True,
        "max_new_tokens": 1000,  
        "pad_token_id": pipeline.tokenizer.eos_token_id,
        }

        predictions = []
        references = [] 
        for row in tqdm.tqdm(ds_predict):

            # exclude examples with empty fields
            if row['main'] == None or row['intro'] == None:
                continue
            
            references.append(row['intro'])
            prompt=prompt_examples + "\n\n" + prompt_template.format(main=row["main"], intro="")[:-1]
            prediction=pipeline(prompt, **gen_config)[0]["generated_text"][len(prompt)+1:]

            if "\n" in prediction:
                prediction=prediction.split("\n")[0]
            predictions.append(prediction)

        scorer = BERTScorer(model_type='bert-base-multilingual-cased')
        P, R, F1 = scorer.score(predictions,references)
        print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        print(results)


if __name__ == "__main__":
    main()
