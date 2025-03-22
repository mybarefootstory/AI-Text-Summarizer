import torch
import gradio as gr

try:
    from transformers import pipeline
except ImportError:
    print("Please install transformers: pip install transformers")
    exit()

try:
    model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")
    text_summary_pipeline = pipeline("summarization", 
                                     model=model_path,
                                     torch_dtype=torch.bfloat16)

    def get_text_summary(input):
        try:
            output = text_summary_pipeline(input)
            return output[0]['summary_text']
        except Exception as e:
            return f"Error in summarization: {str(e)}"

    gr.close_all()
    demo = gr.Interface(fn=get_text_summary, inputs="text", outputs="text",
                        title="Text Summarization AI",  # Add a title here
                        description="Enter text to generate a concise summary") # Optional description)
    demo.launch()

except Exception as e:
    print(f"Error initializing model: {str(e)}")