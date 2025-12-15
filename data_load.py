# Import libraries

import os
from langchain_community.document_loaders import PyPDFLoader

# Define Data Source and load data with PDFLoader

data_load = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
data_test = data_load.load_and_split()
print(len(data_test))
print(data_test[2])
