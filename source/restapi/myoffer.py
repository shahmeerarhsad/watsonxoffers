from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


app = FastAPI()


from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from fastapi import FastAPI  # Importing the FastAPI framework
import uvicorn  # Importing the uvicorn server
from pydantic import BaseModel 

# Import necessary modules from the langchain package
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate
import re
watsonx_API = "77yQYhefKiwxJZS7h4sbbTC9shxdzgqzurEihDwWfJqM" # below is the instruction how to get them
project_id= "3a60480d-5572-4933-899e-9d97cc8bfc6f" # like "0blahblah-000-9999-blah-99bla0hblah0"

app = FastAPI()
class msisdn(BaseModel):
    msisdn: str

@app.get("/")
def read_root():
    return {"Hello": "World!"}

# Root to handle POST request for sum calculation
@app.post("/sum")
def show_offers(numbers: msisdn):
    Tmsisdn = numbers.msisdn
    print(Tmsisdn)
    # Calculating the sum of the provided numbers
    customers = {
        "24091993": {
            "nationality": "American",
            "top_calling_country": "USA",
            "arpu_segment": "High",
            "most_used":"Data"
        },
        "24091994": {
            "nationality": "Indian",
            "top_calling_country": "India",
            "arpu_segment": "Medium",
            "most_used":"Local Mins"
        },
        "24091995": {
            "nationality": "British",
            "top_calling_country": "UK",
            "arpu_segment": "Low",
            "most_used":"Internation Mins"
        }
    }
    best_offers_rules={
        "min_price": "0",
        "max_price": "10",
        "currency": "QAR",
        "min_validity": "0",
        "max_validity":"7",
        "validity_unit":"Days",
        "minimum_data_benefit_unit":"1",
        "maximum_data_benefit_unit":"1024",
        "data_benefit_unit":"MB"
    }

    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : watsonx_API
    }
    params = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }


    PromptTemplate = """ 
        Provide the best possible telecom offers based on the customer attributes such as :
        
        nationality : {nationality}
        arpu segment: {arpu_seg} 
        top calling country : {tcp}
        most_used: {which_data}
        
        Consider the following offer related information while generating the best offer for the customer:
        
        min_price:     {min_price}  
        max_price:     {max_price}    
        currency:      {currency} 
        min_validity:  {min_validity}
        max_validity:  {max_validity}
        validity_unit: {validity_unit}
        minimum_data_benefit_unit:{minimum_data_benefit_unit}
        maximum_data_benefit_unit:{maximum_data_benefit_unit}
        data_benefit_unit:{data_benefit_unit}
        
        
        Be strict to return on following parameters. No other text is required in any way or form.
        Offers = [
        Offer Text: Offer text to be presented to the customer
        Benefit Type : this would be the type of benefit we are recommending
        Benefit Quantity: this would be the amount of receommended benefit
        Price: Price of benefit
        Validity: Validity of benefit.
        ]

        """

    # Set up the LLAMA2 model with the specified parameters and credentials
    LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-405b-instruct',
        credentials=credentials,
        params=params,
        project_id=project_id)

    # Create a Watson LLM instance with the LLAMA2 model
    LLAMA2_model_llm = WatsonxLLM(model=LLAMA2_model)

    prompt= ChatPromptTemplate.from_template(PromptTemplate)
    chain = prompt | LLAMA2_model_llm
    result = chain.invoke({
        "nationality": customers[Tmsisdn]['nationality']
        , "arpu_seg": customers[Tmsisdn]['arpu_segment']
        , "tcp": customers[Tmsisdn]['top_calling_country']
        , "which_data": customers[Tmsisdn]['most_used']
        ,"min_price":  best_offers_rules['min_price']
        ,"max_price":    best_offers_rules['max_price']
        ,"currency":     best_offers_rules['currency']
        ,"min_validity": best_offers_rules['min_validity']
        ,"max_validity": best_offers_rules['max_validity']
        ,"validity_unit":best_offers_rules['validity_unit']
        ,"minimum_data_benefit_unit":best_offers_rules['minimum_data_benefit_unit']
        ,"maximum_data_benefit_unit":best_offers_rules['maximum_data_benefit_unit']
        ,"data_benefit_unit": best_offers_rules['data_benefit_unit']
    })

    # Use regular expression to extract the content between Offers = [ and ]
    pattern = r'Offers = \[(.*?)\]'
    match = re.search(pattern, result, re.DOTALL)

    if match:
        extracted_content = match.group(1).strip()
        print(extracted_content)
        return (extracted_content)
    else:
        return("No match found")


# Running the FastAPI application using uvicorn server
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)