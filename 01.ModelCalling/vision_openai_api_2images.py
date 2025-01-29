import base64
from openai import OpenAI

###################################################################################
## THIS CODE DOESN'T WORK WITH llama3.2-vision, THE CALL TO THE MODEL NEVER RETURNS
###################################################################################

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
        model="minicpm-v",
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
        # temperature=0.8
    )

    return response


response = call_model("From which country are these two photos from? Use whatever visual cues you have, like signs and its text, clothing, skin colour, architecture styles, to make a deduction.", "image6.jpg", "image3.png")
model_output = response.choices[0].message.content
print(model_output)

print("Total tokens:", response.usage.total_tokens, "\n")

# Correct response, even if it didn't identify the bridge:
# 
# ****** For image image6.jpg & image3.png, asking: From which country are these two photos from? Use whatever visual cues you have, like signs and its text, clothing, skin colour, architecture styles, to make a deduction.
# The first photo shows a bridge with illuminated cables at night. The structure appears modern, likely indicating a developed area within Europe due to the style of construction which is common in countries such as Germany or Scandinavian nations where engineering and infrastructure are well maintained during nighttime lighting for safety reasons and aesthetic appeal.
#
# The second photo depicts people outside what seems to be an administrative building with signage that includes Portuguese. The architecture has distinctive European features, including stone pavement and a white-walled structure with red-tiled roofs, which is characteristic of Portugal or another neighboring country in Southern Europe.
#
# Given the visible signs and architectural cues such as cobblestone streets and style buildings similar to what are found throughout many parts of southern and western Europe but particularly fitting Portuguese culture where this architecture is prominent. Therefore, it's reasonable to deduce that these photos were taken in Portugal.