**TF-IDF Processing Pipeline **

A full TF-IDF implementation written in Python **with no pre-built TF-IDF libraries**.  
All TF, IDF, and TF-IDF calculations are performed manually using standard Python, as required by the assignment brief.

The project also supports a **configurable preprocessing pipeline**, allowing you to choose whether to apply:

- **c** → cleaning (lowercasing + removing punctuation)  
- **sw** → stopword removal  
- **st** → stemming  

Final output filenames include the preprocessing steps applied, e.g.: tf_idf_c_sw_st.csv

## 📂 **Project Structure**
TF_IDF_project/  
│  
├── main.py # main pipeline controller  
├── data_loader.py # loads corpus from file  
├── processor.py # preprocessing, tokenisation, TF, IDF, TF-IDF logic  
├── data_preview.py # helper for visualising results  
├── data/ # Folder containing source documents
├── outputs/ # folder containing csv outputs  
└── README.md # this file  
