#!/usr/bin/env python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    name.submit(greet, inputs=name, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7861, share=False, debug=True)