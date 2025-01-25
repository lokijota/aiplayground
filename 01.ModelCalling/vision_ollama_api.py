import ollama

images = ["image1.jpg", "image2.jpg", "image3.png", "image4.jpg", "image5.png", "image6.jpg", "image7.png"]

prompts = dict()
prompts["image1.jpg"] = "Can you summarize the content in a single sentence?" 
prompts["image2.jpg"] = "How many pieces of tomato are there on the picture? Explain your reasoning step by step and verify that it is correct, and point out where they are in the photo."
prompts["image3.png"] = "Where do you think this was taken?"
prompts["image4.jpg"] = "Can you guess where that fabric is from, based on the design?"
prompts["image5.png"] = "It's not Insbruck. Try again: what is this?"
prompts["image6.jpg"] = "What is the bridge and in which european country is it located?"
prompts["image7.png"] = "What bitter juice is in the glass? Give me three guesses"

def call_model(prompt, image):

    print(f"****** For image {image}, asking: {prompt}")

    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': f'{prompt}',
            'images': [f'./01.ModelCalling/{image}'],
        }],
        options={
            "temperature": 0.7
        }, 
        stream=False

    )

    return response


for img in images:
    response = call_model("What's in the image?", img)
    model_output = response.message["content"]
    print(model_output)

    print("Total duration (s):", int(response["total_duration"])/1e9, "\n")

    if img in prompts:
        response = call_model(prompts[img], img)
        model_output = response.message["content"]
        print(model_output)

        print("Total duration (s):", int(response["total_duration"])/1e9, "\n")


