import base64
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)


images = ["image1.jpg", "image2.jpg", "image3.png", "image4.jpg", "image5.png", "image6.jpg", "image7.png"]

prompts = dict()
prompts["image1.jpg"] = "Can you summarize the content in a single sentence?" 
prompts["image2.jpg"] = "How many pieces of tomato are there on the picture? Explain your reasoning step by step and verify that it is correct, and point out where they are in the photo."
prompts["image3.png"] = "Where do you think this was taken?"
prompts["image4.jpg"] = "Can you guess where that fabric is from, based on the design?"
prompts["image5.png"] = "It's not Insbruck. Try again: what is this?"
prompts["image6.jpg"] = "What is the bridge and in which european country is it located?"
prompts["image7.png"] = "What bitter juice is in the glass? Give me three guesses"

# includes sample code from: https://platform.openai.com/docs/guides/vision?lang=curl

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def call_model(prompt, image):

    print(f"****** For image {image}, asking: {prompt}")

    # read image and convert to base64
    if image.endswith("jpg"):
        ct = "image/jpeg"
    else:
        ct = "image/png"

    base64_image = encode_image(f'./01.ModelCalling/{image}')
    image_url_param = f"data:{ct};base64,{base64_image}"

    response = client.chat.completions.create(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_param},
                    },
                ],
            }
        ],
        temperature=0.8
    )

    return response



for img in images:
    response = call_model("What's in the image?", img)
    model_output = response.choices[0].message.content
    print(model_output)

    print("Total tokens:", response.usage.total_tokens, "\n")

    if img in prompts:
        response = call_model(prompts[img], img)
        model_output = response.choices[0].message.content
        print(model_output)

        print("Total tokens:", response.usage.total_tokens, "\n")


