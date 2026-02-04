from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer


data_i =r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\data_test\base_data"
key = "text"
opt = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)\output_test"
tok_pth = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Tokenizer\data\tokenizer\tokenizer.json"
n_task = 2
end = '<|eos|>'
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
                batch_size = 100,
                shuffle_documents = False,
                max_tokens_per_file=1e9,
            ),
        ],
        tasks=n_task,
        workers= 6,
        start_method='spawn', 
    )
    preprocess_executor.run()


if __name__ == "__main__":
    main()
