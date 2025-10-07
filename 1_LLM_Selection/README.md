## MediPal -- LLM testing and selection

### In this section, I set five tasks to test three LLMs' performance, so that I can understand them and design the application properly.
##### Tasks:
1. Reasoning
2. Generate questions based on given content
3. Answer question based on given content
4. Solve Yes or No question
5. Structured ouput

##### Technique

1. CoT prompting
2. A few shots prompting

##### Target LLMs:
1. meta-llama/Llama-3.2-1B-Instruct 
2. meta-llama/Meta-Llama-3-8B-Instruct
3. ContactDoctor/Bio-Medical-Llama-3-8B

##### Why test Bio-Medical-Llama-3-8B？

Bio-Medical-Llama-3-8B model is a specialized large language model designed for biomedical applications. It is finetuned from the meta-llama/Meta-Llama-3-8B-Instruct model using a custom dataset containing over 500,000 diverse entries. These entries include a mix of synthetic and manually curated data, ensuring high quality and broad coverage of biomedical topics.

The model is trained to understand and generate text related to various biomedical fields, making it a valuable tool for researchers, clinicians, and other professionals in the biomedical domain.

### Results:

#### Task 1: Reasoning

| LLM                                   | LetterTest | Simple math  | Harder math   |
|---------------------------------------|------------------|--------------------|--------------|
| meta-llama/Llama-3.2-1B-Instruct      | Resoining👍, Result👎 | Resoining👎, Result👎 | Resoining👎, Result👎 |
| meta-llama/Meta-Llama-3-8B-Instruct   | Resoining👍, Result👍 | Resoining👍, Result👍 | Resoining👍, Result👍 |
| ContactDoctor/Bio-Medical-Llama-3-8B  | Resoining👎, Result👎 | Resoining👍, Result👍 | Resoining👌, Result👎 |


#### Task 2: Generate questions based on given content
| LLM                                   | Question Quality | Structured_output  | 
|---------------------------------------|------------------|--------------------|
| meta-llama/Llama-3.2-1B-Instruct      | Bad👎            | Bad👎              | 
| meta-llama/Meta-Llama-3-8B-Instruct   | Good👌           | Amazing👍          | 
| ContactDoctor/Bio-Medical-Llama-3-8B  | Amazing👍        | Good👌              | 

* Note: The questions Bio-Medical-Llama-3-8B generates are better because they sound more like natural user input rather than formal written language.

#### Task 3: Answer question based on given content
| LLM                                   | Answer Quality | Structured_output  | 
|---------------------------------------|------------------|--------------------|
| meta-llama/Llama-3.2-1B-Instruct      | Good👌            | 👍             | 
| meta-llama/Meta-Llama-3-8B-Instruct   | Amazing👍        | 👍             | 
| ContactDoctor/Bio-Medical-Llama-3-8B  | Good👌            | 👍             | 

#### Task 4: Solve Yes or No question
| LLM                                   | Result | Structured_output  | 
|---------------------------------------|------------------|--------------------|
| meta-llama/Llama-3.2-1B-Instruct      | Pretty Bad👎     | 👍             | 
| meta-llama/Meta-Llama-3-8B-Instruct   | Amazing👍        | 👍             | 
| ContactDoctor/Bio-Medical-Llama-3-8B  | Good👌           | 👍             | 


#### Task 5: Structured ouput
| LLM                                   | Result | Structured_output  | 
|---------------------------------------|------------------|--------------------|
| meta-llama/Llama-3.2-1B-Instruct      | Bad👎            | Bad👎             | 
| meta-llama/Meta-Llama-3-8B-Instruct   | Amazing👍        | Amazing👍         | 
| ContactDoctor/Bio-Medical-Llama-3-8B  | Amazing👍        | Amazing👍         | 

### Summary

* meta-llama/Llama-3.2-1B-Instruct: Fails in most of tasks, only answer question is good.
* meta-llama/Meta-Llama-3-8B-Instruct: Best overall in reasoning, accuracy, and structure.
* ContactDoctor/Bio-Medical-Llama-3-8B: Does good job overall, the only capability is better than Meta-Llama-3-8B-Instruct is generating questions based on given content.

So I will let meta-llama/Meta-Llama-3-8B-Instruct handle most tasks. ContactDoctor/Bio-Medical-Llama-3-8B will take the task where need to generate content based on give medicine content.