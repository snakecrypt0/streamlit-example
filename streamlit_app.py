import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from twit import tweeter
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection

import streamlit as st

load_dotenv()

# 1. Tool for search


def search(query):
    # Define the URL for the SERPHouse API
    url = "https://api.serphouse.com/serp/live"

    # Define the payload with the search query and other parameters
    payload = json.dumps({
        "data": {
            "q": query,
            "domain": "google.com",
            "loc": "Abernathy,Texas,United States",
            "lang": "en",
            "device": "desktop",
            "serp_type": "web",
            "page": "1",
            "verbatim": "0"
        }
    })

    # Define the headers with the API key and content type
    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        'authorization': "Bearer " + os.getenv("serphouse_api_key")
    }

    # Make the POST request to the SERPHouse API
    response = requests.request("POST", url, data=payload, headers=headers)

    # Print the response text
    print(response.text)

    # Return the response text
    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    
    # Define the webdriver for Bright Data
    SBR_WEBDRIVER = 'https://brd-customer-hl_41b06450-zone-scraping_browser:m08biamyzwv3@brd.superproxy.io:9515'
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
    
    # Use Selenium to navigate to the website and scrape the content
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        print('Connected! Navigating to the website...')
        driver.get(url)
        print('Navigated! Scraping page content...')
        html = driver.page_source
        print(html)

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    print("CONTENTTTTTT:", text)

    # Convert the text to JSON format
    json_text = json.dumps(text)

    # If the content is too large, summarize it
    if len(json_text) > 10000:
        output = summary(objective, json_text)
        return output
    else:
        return json_text



def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
    will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact
    Do not describe what the webpage is, you are here to get acurate and specific information
    Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
    Here is the text:

    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ Always look at the web first
            7/ Output as much information as possible, make sure your answer is at least 500 WORDS
            8/ Be specific about your reasearch, do not just point to a website and say things can be found here, that what you are for
            

            Example of what NOT to do return these are just a summary of whats on the website an nothing specific, these tell the user nothing!!

            1/WIRED - WIRED provides the latest news, articles, photos, slideshows, and videos related to artificial intelligence. Source: WIRED

            2/Artificial Intelligence News - This website offers the latest AI news and trends, along with industry research and reports on AI technology. Source: Artificial Intelligence News
            """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


template = """
    You are a very experienced ghostwriter who excels at writing Twitter threads.
You will be given a bunch of info below and a topic headline, your job is to use this info and your own knowledge
to write an engaging Twitter thread.
The first tweet in the thread should have a hook and engage with the user to read on.

Here is your style guide for how to write the thread:
1. Voice and Tone:
Informative and Clear: Prioritize clarity and precision in presenting data. Phrases like "Research indicates," "Studies have shown," and "Experts suggest" impart a tone of credibility.
Casual and Engaging: Maintain a conversational tone using contractions and approachable language. Pose occasional questions to the reader to ensure engagement.
2. Mood:
Educational: Create an atmosphere where the reader feels they're gaining valuable insights or learning something new.
Inviting: Use language that encourages readers to dive deeper, explore more, or engage in a dialogue.
3. Sentence Structure:
Varied Sentence Lengths: Use a mix of succinct points for emphasis and longer explanatory sentences for detail.
Descriptive Sentences: Instead of directive sentences, use descriptive ones to provide information. E.g., "Choosing a topic can lead to..."
4. Transition Style:
Sequential and Logical: Guide the reader through information or steps in a clear, logical sequence.
Visual Emojis: Emojis can still be used as visual cues, but opt for ones like ℹ️ for informational points or ➡️ to denote a continuation.
5. Rhythm and Pacing:
Steady Flow: Ensure a smooth flow of information, transitioning seamlessly from one point to the next.
Data and Sources: Introduce occasional statistics, study findings, or expert opinions to bolster claims, and offer links or references for deeper dives.
6. Signature Styles:
Intriguing Introductions: Start tweets or threads with a captivating fact, question, or statement to grab attention.
Question and Clarification Format: Begin with a general question or statement and follow up with clarifying information. E.g., "Why is sleep crucial? A study from XYZ University points out..."
Use of '➡️' for Continuation: Indicate that there's more information following, especially useful in threads.
Engaging Summaries: Conclude with a concise recap or an invitation for further discussion to keep the conversation going.
Distinctive Indicators for an Informational Twitter Style:

Leading with Facts and Data: Ground the content in researched information, making it credible and valuable.
Engaging Elements: The consistent use of questions and clear, descriptive sentences ensures engagement without leaning heavily on personal anecdotes.
Visual Emojis as Indicators: Emojis are not just for casual conversations; they can be effectively used to mark transitions or emphasize points even in an informational context.
Open-ended Conclusions: Ending with questions or prompts for discussion can engage readers and foster a sense of community around the content.

Last instructions:
The twitter thread should be between the length of 3 and 10 tweets 
Each tweet should start with (tweetnumber/total length)
Dont overuse hashtags, only one or two for entire thread.
Use links sparingly and only when really needed, but when you do make sure you actually include them! 
Only return the thread, no other text, and make each tweet its own paragraph.
Make sure each tweet is lower that 220 chars
    Topic Headline:{topic}
    Info: {info}
    """

prompt = PromptTemplate(
    input_variables=["info","topic"], template=template
)

llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    
)


twitapi = tweeter()

def tweetertweet(thread):

    tweets = thread.split("\n\n")
   
    #check each tweet is under 280 chars
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm.predict(prompt)[:280]
    #give some spacing between sentances
    tweets = [s.replace('. ', '.\n\n') for s in tweets]

    for tweet in tweets:
        tweet = tweet.replace('**', '')

    response = twitapi.create_tweet(text=tweets[0])
    id = response.data['id']
    tweets.pop(0)
    for i in tweets:
        print("tweeting: " + i)
        reptweet = twitapi.create_tweet(text=i, 
                                    in_reply_to_tweet_id=id, 
                                    )
        id = reptweet.data['id']


  

def main():
    # Set page title and icon
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    # Display header 
    st.header("AI research agent :bird:")
    
    # Get user's research goal input
    query = st.text_input("Research goal")

    # Initialize result and thread state if needed
    if not hasattr(st.session_state, 'result'):
        st.session_state.result = None

    if not hasattr(st.session_state, 'thread'):
        st.session_state.thread = None

    # Do research if query entered and no prior result
    if query and (st.session_state.result is None or st.session_state.thread is None):
        st.write("Doing research for ", query)

        # Run agent to generate result
        st.session_state.result = agent({"input": query})
        
        # Generate thread from result
        st.session_state.thread = llm_chain.predict(topic=query, info=st.session_state.result['output'])

    # Display generated thread and result if available
    if st.session_state.result and st.session_state.thread:
        st.markdown(st.session_state.thread)
        
        # Allow tweeting thread
        tweet = st.button("Tweeeeeet")
        
        # Display info on result 
        st.markdown("Twitter thread Generated from the below research")
        st.markdown(st.session_state.result['output'])
    
        if tweet:
            # Tweet thread
            tweetertweet(st.session_state.thread)
            
 

if __name__ == '__main__':
    main()

