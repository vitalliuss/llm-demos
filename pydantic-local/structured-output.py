import openai
from pydantic import BaseModel, Field

client = openai.Client(
    base_url="http://localhost:1234/v1/"
)

custom_prompt = ('''You are review analyzing agent. Your goal is to tag the review. Read the review below, analyze and 
                 tag by the scale from 0 to 5 how this affects each category. If you don't have enough data use 0. 1 
                 is bad, 2 is so-so, 3 is ok, 4 is good, 5 is excellent. The categories are: price, quality, 
                 delivery, customer service, and overall satisfaction.
                 ''')


class ReviewTagged(BaseModel):
    id: int = Field(description="The ID of the review")
    review: str = Field(description="The review text")
    price: int = Field(description="The price rating")
    product: int = Field(description="The product rating")
    delivery: int = Field(description="The delivery rating")
    customer_service: int = Field(description="The customer service rating")
    overall_satisfaction: int = Field(description="The overall satisfaction rating")
    insufficient_data: bool = Field(description="If the model does not have enough data to make a review tagging")


def analyze_feedback(review, custom_prompt):
    response = client.beta.chat.completions.parse(
        model="phi-4",
        messages=[
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": review}
        ],
        temperature=0.3,
        response_format=ReviewTagged,
    )
    return response.choices[0].message.parsed


if __name__ == '__main__':
    reviews = [
        'The product was great, but the delivery was slow. The customer service was helpful, but the price was too high.',
        'The product was terrible, but the delivery was fast. The customer service was unhelpful, but the price was low.',
        'The product was okay, but the delivery was late. The customer service was okay, but the price was high.',
        'The product was good, but the delivery was on time. The customer service was excellent, but the price was reasonable.',
        'Can I just have a slice of pizza? Thanks.']
    tagged_reviews = []
    for review in reviews:
        tag_info = analyze_feedback(review, custom_prompt)
        tagged_reviews.append(tag_info)

    print(tagged_reviews)