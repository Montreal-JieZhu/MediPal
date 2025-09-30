import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,          # the function
    inputs="text",     # text box input
    outputs="text"     # text output
)

demo.launch()

# python test_gradio.py
