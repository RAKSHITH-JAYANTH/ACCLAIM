# ACCLAIM
Long Context LLM Acceleration on Heterogeneous Edge Platforms 


<h4> Installation Requirements: </h4>

<item> Torch </item>

<item> Numpy </item>

<item> Json </item>

<item> Intel extension for pytorch </item>

<item> PyYAML </item>

<item> jinja2 </item>

<item> tiktoken </item>

<item> pydantic>=2 </item>

<item> Pillow </item>

<h4> Code Details </h4>

The implementation of the code for Llama models is in  the path ".llama_all\llama_models1\llama3\reference_impl\".
The implementation of the code for Gemma models is in the path ".gemma_all\gemma\"

File model.py contains the  model implementation.

The code provides the performance details of each segment of the code and finally displays the performance in TTFT.


<h4> Instructions to Run </h4>

1. From the Llama website, download the Llama3.2 - 1B and 3B models - [Download Llama](https://www.llama.com/llama-downloads/)
2. From the Kaggle website, download the Gemma2 - 2B model - [Download Gemma](https://www.kaggle.com/models/google/gemma-2/pyTorch/gemma-2-2b-it/) 

3. Clone the repository

4. Create a folder and copy the downloaded model and checkpoint files into the folder
5. Run - "python run.py model_dir --max_seq_len length" :- (Preferred lengths: 8K, 16K, 32K ..)
6. The current code has a predefined very long prompt (about 300K tokens). A subset of tokens (8K, 16K etc) is used for analysis.









