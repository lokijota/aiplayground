import base64
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

images = ["image1.jpg", "image2.jpg", "image3.png", "image4.jpg", "image5.png", "image6.jpg", "image7.png"]

# includes sample code from: https://platform.openai.com/docs/guides/vision?lang=curl

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def call_model(prompt, image1, image2):

    print(f"****** For image {image1} & {image2}, asking: {prompt}")

    # read image and convert to base64
    if image1.endswith("jpg"):
        ct1 = "image/jpeg"
    else:
        ct1 = "image/png"

    base64_image1 = encode_image(f'./01.ModelCalling/{image1}')
    image_url_param1 = f"data:{ct1};base64,{base64_image1}"

    # read image and convert to base64
    if image2.endswith("jpg"):
        ct2 = "image/jpeg"
    else:
        ct2 = "image/png"

    base64_image2 = encode_image(f'./01.ModelCalling/{image2}')
    image_url_param2 = f"data:{ct2};base64,{base64_image2}"


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
                        "image_url": {"url": image_url_param1},
                    },
                                        {
                        "type": "image_url",
                        "image_url": {"url": image_url_param2},
                    },
                ],
            }
        ],
        temperature=0.8
    )

    return response


response = call_model("What could there be in common in these images? Both in the visual composition but other aspects, like where they were most taken, etc?",\
                       "image3.png", "image6.jpg")
model_output = response.choices[0].message.content
print(model_output)

print("Total tokens:", response.usage.total_tokens, "\n")