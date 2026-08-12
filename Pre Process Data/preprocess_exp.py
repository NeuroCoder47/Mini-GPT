from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer

split = "test"   

if split == "train":
    n_task_tr= 10
    workers_tr = 10
    data_i = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\data_train"
    opt    = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\output_train"
else:
    n_task_tr= 1
    workers_tr = 1
    data_i = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\data_test"
    opt    = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\output_test"

tok_pth = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Tokenizer\tokenizer.json"
end = '<|eos|>'
n_task = n_task_tr
key = "text"

def main():
    datatrove_reader = ParquetReader(
        data_folder=data_i, 
        text_key=key,
        glob_pattern="*.parquet"  
    )
    
    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            datatrove_reader,
            DocumentTokenizer(
                output_folder=opt,
                tokenizer_name_or_path=tok_pth,
                eos_token=end,
                batch_size = 1000,
                shuffle_documents = False,
                max_tokens_per_file=1e9,
            ),
        ],
        tasks=n_task,
        workers= workers_tr,
        start_method='spawn', # This is for windows for linux it will be different 
    )
    preprocess_executor.run()


if __name__ == "__main__":
    main()
# You are importing the LocalPipelineExecutor to run the pipeline on your local machine.
# You are importing ParquetReader to read .parquet files as input data.
# You are importing DocumentTokenizer to convert raw text into tokens.

# data_i is the folder path where your raw .parquet files are stored.
# key is the column name in the parquet file that contains the actual text.
# opt is the folder path where the tokenized output files will be saved.
# tok_pth is the path to your custom tokenizer file (tokenizer.json).
# n_task = 2 means your data will be split into 2 chunks and processed separately.
# end is the special token that will be added at the end of every single document.

# Inside main() you are building and running the full pipeline.

# ParquetReader goes into your data_i folder.
# It scans for all files matching *.parquet.
# For each row in those files, it reads the "text" column as one Document.
# Each Document then enters the pipeline and flows to the next step.

# LocalPipelineExecutor is the thing that actually runs everything.
# You give it a list of steps (the pipeline) and it executes them in order.

# The pipeline has two steps:
# Step 1 is datatrove_reader — it reads and produces documents.
# Step 2 is DocumentTokenizer — it receives each document and tokenizes it.

# DocumentTokenizer takes each document's text and converts it into token IDs.
# It uses your custom tokenizer.json file to do the conversion.
# After tokenizing each document, it appends the '<|eos|>' token at the very end.
# It processes 100 documents at a time because batch_size is set to 100.
# shuffle_documents = False means the order of documents is preserved as-is.
# max_tokens_per_file = 1e9 means each output file can hold up to 1 billion tokens.
# Once a file hits 1 billion tokens, a new output file is started automatically.

# tasks = 2 means the data is divided into 2 chunks.
# workers = 6 means up to 6 tasks can run at the same time on your machine.
# But since tasks = 2, only 2 will ever run at once even though 6 workers are available.
# start_method = 'spawn' is required on Windows for multiprocessing to work correctly.
# On Linux the default fork method works, so you would not need to set this.

# preprocess_executor.run() is what actually starts the whole process.
# It kicks off the reader, streams documents into the tokenizer, and saves the output.