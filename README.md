# AcLLM
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

The implementation of the code is in  the path ".\llama_models1\llama3\reference_impl\" 

Files model.py and generation.py contain the model implementation.

The code provides the performance details of each segment of the code and finally displays the performance in TTFT.


<h4> Instructions to Run </h4>

1. From the llama website, download the Llama3.2 - 1B model - [Download Llama](https://www.llama.com/llama-downloads/)

2. Clone the repository

3. Create a folder (say, Llama3.2_1B) and copy the downloaded model and checkpoint files into the folder
4. Run - "python example_test_completion.py model_dir --max_seq_len length" :- (Preferred lengths: 8K, 16K, 32K ..)
5. The current code has a predefined very long prompt (about 300K tokens). A subset of tokens (8K, 16K etc) is used for analysis.









