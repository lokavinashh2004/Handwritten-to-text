from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import gradio as gr
import spaces

# Initialize model and processor
ckpt = "unsloth/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16
).to("cuda")
processor = AutoProcessor.from_pretrained(ckpt)

@spaces.GPU
def extract_text(image):
    # Convert image to RGB
    image = Image.open(image).convert("RGB")
    
    # Create message structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract handwritten text from the image and output only the extracted text without any additional description or commentary in output"},
                {"type": "image"}
            ]
        }
    ]
    
    # Process input
    texts = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=texts, images=[image], return_tensors="pt").to("cuda")

    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=250)
    result = processor.decode(outputs[0], skip_special_tokens=True)

    print(result)
    
    # Clean up the output to remove the prompt and assistant text
    if "assistant" in result.lower():
        result = result[result.lower().find("assistant") + len("assistant"):].strip()
    
    # Remove any remaining conversation markers
    result = result.replace("user", "").replace("Extract handwritten text from the image and output only the extracted text without any additional description or commentary in output", "").strip()

    print(result)
    
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=extract_text,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(label="Extracted Text"),
    title="Handwritten Text Extractor",
    description="Upload an image containing handwritten text to extract its content.",
)

# Launch the app
demo.launch(debug=True)