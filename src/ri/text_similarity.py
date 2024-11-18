from sentence_transformers import SentenceTransformer, util
st_model = SentenceTransformer('all-MiniLM-L6-v2')


res_dict = {
    "Outline a plan to get information from website": '''To get information from a website, you can follow a structured plan. This plan can be adapted depending on the type of website, the data you need, and the tools you have available. Here's a general outline:

1. Define Objectives
Identify the Data: Determine what information you need from the website (e.g., product details, news articles, user reviews).
Purpose of Data: Clarify why you need this data and how you plan to use it.
2. Understand the Website
Website Structure: Analyze the structure of the website (HTML, CSS, JavaScript).
Identify Key Pages: Identify the pages where the desired information is located.
Check for APIs: See if the website provides an API that can be used to retrieve data more efficiently.
3. Choose Tools and Technologies
Web Scraping Tools: Select a tool or library for scraping, such as:
Python Libraries: BeautifulSoup, Scrapy, Requests
Browser Automation Tools: Selenium, Puppeteer
Data Storage: Choose a storage solution for the data, like CSV files, databases (SQL or NoSQL), or spreadsheets.''',



    "Write a tutorial on how to make a toy using household items": '''Creating a toy from household items can be a fun and creative project. Here's a step-by-step tutorial on how to make a simple, enjoyable toy: a Homemade Tin Can Telephone. This toy is easy to make and provides an engaging experience for children as they learn about sound transmission.

Materials Needed:
Two empty tin cans (e.g., soup cans or coffee cans)
A length of string or yarn (about 10-20 feet)
A nail and hammer (or a screwdriver)
Scissors
Optional: Paints, stickers, or markers (for decorating)
Steps to Make the Tin Can Telephone:
1. Prepare the Cans
Clean and Remove Labels: Ensure the cans are clean and free of labels. Wash them thoroughly and remove any adhesive residue.
Punch Holes: Using a nail and hammer (or a screwdriver), carefully punch a small hole in the center of the bottom of each can. Make sure the hole is just big enough for the string to pass through.
2. Prepare the String
Cut the String: Cut a length of string or yarn to your desired length. For a basic tin can telephone, 10-20 feet is ideal. You can always trim it later if needed.
3. Attach the String to the Cans
Thread the String: Push one end of the string through the hole in one of the cans. Tie a knot on the inside of the can to secure the string in place. Repeat this step for the other can.
4. Test Your Tin Can Telephone
Position the Cans: Hold one can up to your ear and have someone speak into the other can. You may need to pull the string taut to get clear sound transmission.

Adjust Length: If you notice that the sound is not clear, make sure the string is tight and there are no kinks or knots in it.

5. Decorate (Optional)
Personalize Your Toy: Use paints, stickers, or markers to decorate the outside of the cans. This can make the toy more visually appealing and fun to use.''',
}

questions = [
    ("Outline a plan to get information from website"),
    ("Write a tutorial on how to make a toy using household items"),
]


def calculate_text_similarity(text1: str, text2: str) -> float:
    embeddings = st_model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def get_similar_text(text_ori: str):
    question_ = questions[0]
    similarity_ = calculate_text_similarity(text_ori, question_)
    
    for question in questions:
        similarity = calculate_text_similarity(text_ori, question)
        if similarity_ < similarity:
            similarity_ = similarity
            question_ = question
    # TODO return similarity
    return question_


def get_similar_res(text):
    return res_dict[text]