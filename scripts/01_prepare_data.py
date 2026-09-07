import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
from utils import get_names, string_cleaning, punctuation_ratio, ensure_exists_dirs, word_count_and_shortend, capitalize_first
import pandas as pd
import subprocess
from faker import Faker
import xml.etree.ElementTree as ET
import random
import re
import requests
import wget
import glob
import gzip
import json 
from tqdm import tqdm

def prepare_s2orc_ACL(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    CACHE_PATH_S2ORC = os.path.join(CACHE_PATH, "S2ORC")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_DATASET = os.path.join(ARTEFACT_PATH, "DATASET.txt")
    ensure_exists_dirs([CACHE_PATH, CACHE_PATH_S2ORC, DATASET_PATH, ARTEFACT_PATH, PREPARED_DATA_PATH])
    random.seed(cfg.run.seed)    

    headers = {
        "x-api-key": cfg.dataset.api_key
    }
    request_dataset = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{cfg.dataset.release_id}/dataset/s2orc", 
                      headers=headers).json()
    
    print("Download datset shards")
    shard_ids = []
    for url in tqdm(request_dataset["files"]):
        match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
        assert match.group(1) == cfg.dataset.release_id
        shard_id = match.group(2)
        shard_ids.append(shard_id)
        if not os.path.isfile(os.path.join(CACHE_PATH_S2ORC, f"{shard_id}.gz")):
            wget.download(url, out=os.path.join(CACHE_PATH_S2ORC, f"{shard_id}.gz"))


    print("Parse S2ORC files.")
    dataset_dict = {
        "id": [],
        "abstract": [],
        "introduction": [],
        "title": [], 

    }

    # This pattern matches the old ACL pattern ID format
    # PDNC gives the main ACL venue, D=EMNLP, N=NAACL, C=COLING
    # 09 - 18 matches the years 2009-2018
    # the 1 after the dash indicates long papers. 
    # The last 3 digits are the unique id of the paper.
    venue_pattern = re.compile(r'^[PDNC](09|1[0-8])-1\d{3}$')
    for shard_id in tqdm(shard_ids):
        with gzip.open(os.path.join(CACHE_PATH_S2ORC, f"{shard_id}.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                paper = json.loads(line)
                # Check if the paper comes with external ids
                if paper["externalids"]:
                    # Check if the paper has an ACL attribute
                    if "acl" in paper["externalids"]:
                        # Check if the ACL attribute is not empty
                        if paper["externalids"]["acl"]:
                            if venue_pattern.match(paper["externalids"]["acl"]):
                                try:
                                    abstract_borders = eval(paper["content"]["annotations"]["abstract"])
                                    abstract_parts = []
                                    for abstract_border in abstract_borders:
                                        abstract_parts.append(paper['content']['text'][int(abstract_border["start"]): int(abstract_border["end"])])
                                    abstract = "\n".join(abstract_parts)

                                    section_headers = eval(paper["content"]["annotations"]["sectionheader"])
                                    paragraph_borders = eval(paper["content"]["annotations"]["paragraph"])
                                    first_section_header = section_headers[0]
                                    # Check if the paper has an introduction setting 
                                    if paper['content']['text'][int(first_section_header["start"]): int(first_section_header["end"])] == "Introduction":
                                        first_section_end = section_headers[1]["start"]
                                        introduction_parts = []
                                        # Get all paragraphs of the first chapter
                                        for paragraph_border in paragraph_borders:
                                            if paragraph_border["end"] < first_section_end:
                                                introduction_parts.append(paper["content"]["text"][int(paragraph_border["start"]): int(paragraph_border["end"])])
                                            else: 
                                                break
                                    introduction = "\n".join(introduction_parts)
                                    
                                    if paper["content"]["annotations"]["title"]:
                                        title_borders = eval(paper["content"]["annotations"]["title"]) 
                                        title_parts  = []
                                        for title_border in title_borders:
                                            title_parts.append(paper["content"]["text"][int(title_border["start"]): int(title_border["end"])])
                                        title = "".join(title_parts)
                                    else:
                                        title = ""

                                    if abstract != "" and introduction != "" and title != "":
                                        dataset_dict["id"].append(paper["externalids"]["acl"])
                                        dataset_dict["abstract"].append(abstract)
                                        dataset_dict["introduction"].append(introduction)
                                        dataset_dict["title"].append(title)
                                except Exception as e:
                                    print(e)
    dataset = pd.DataFrame(dataset_dict).set_index("id")
    print(dataset)
    print("Clean dataset")
    dataset[cfg.dataset.text_name] = dataset[cfg.dataset.text_name].map(string_cleaning)
    dataset["abstract"] = dataset["abstract"].map(string_cleaning)
    dataset["introduction"] = dataset["introduction"].map(string_cleaning)
    dataset["title"] = dataset["title"].map(string_cleaning)

    print("Spacy tag the dataset")
    word_counts, texts_shortend = word_count_and_shortend(dataset[cfg.dataset.text_name].astype(str).to_list(), max_words = cfg.run.min_len_tokens, cap = cfg.run.token_cap)
    
    n = len(dataset)
    dataset[cfg.dataset.text_name + "_word_count"] = word_counts
    dataset[cfg.dataset.text_name] = texts_shortend 

    dataset[dataset[cfg.dataset.text_name + "_word_count"] < cfg.run.min_len_tokens].to_csv(os.path.join(ARTEFACT_PATH, cfg.dataset.name + "_story_too_short.csv"))
    dataset = dataset[dataset[cfg.dataset.text_name + "_word_count"] >= cfg.run.min_len_tokens]
    removed_too_short_text = n - len(dataset)
    print("Save dataset")
    dataset.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"))

    with open(DOCUMENTATION_FILE_DATASET, "w") as f:
        f.write(f"Removed {removed_too_short_text} because the text was shorter than {cfg.run.min_len_tokens}.\n")



def prepare_wikiHow(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_DATASET = os.path.join(ARTEFACT_PATH, "DATASET.txt")
    ensure_exists_dirs([CACHE_PATH, DATASET_PATH, ARTEFACT_PATH, PREPARED_DATA_PATH])
    random.seed(cfg.run.seed)

    # Extract the bnc2014 spoken dataset into the cache
    if os.path.isfile(os.path.join(cfg.datapath, "wikihowAll.csv.zip")):
        if not os.path.isdir(os.path.join(CACHE_PATH, "wikihowAll")):
            print("Unzipping the wikiHow dataset")
            subprocess.run(["unzip", os.path.join(cfg.datapath, "wikihowAll.csv.zip"),
                        "-d", os.path.join(CACHE_PATH, "wikihowAll")])
    else:
        print(f"Please put the wikiHow dataset as a zip at: {os.path.join(cfg.datapath, 'wikihowAll.csv.zip')}")

    dataset = pd.read_csv(os.path.join(CACHE_PATH, "wikihowAll", "wikihowAll.csv"))
    
    n = len(dataset)
    dataset = dataset.dropna()
    dropped_nan = n - len(dataset)

    print("Clean dataset")
    dataset[cfg.dataset.text_name] = dataset[cfg.dataset.text_name].map(string_cleaning)
    dataset["title"] = dataset["title"].map(string_cleaning)
    dataset["headline"] = dataset["headline"].map(string_cleaning)

    print("Spacy tag the dataset")
    word_counts, texts_shortend = word_count_and_shortend(dataset[cfg.dataset.text_name].astype(str).to_list(), max_words = cfg.run.min_len_tokens, cap = cfg.run.token_cap)
    
    n = len(dataset)
    dataset[cfg.dataset.text_name + "_word_count"] = word_counts
    dataset[cfg.dataset.text_name] = texts_shortend 

    dataset[dataset[cfg.dataset.text_name + "_word_count"] < cfg.run.min_len_tokens].to_csv(os.path.join(ARTEFACT_PATH, cfg.dataset.name + "_story_too_short.csv"))
    dataset = dataset[dataset[cfg.dataset.text_name + "_word_count"] >= cfg.run.min_len_tokens]
    removed_too_short_text = n - len(dataset)
    print("Save dataset")
    dataset.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"))

    with open(DOCUMENTATION_FILE_DATASET, "w") as f:
        f.write(f"Removed {dropped_nan} because they contained nan columns.\n")
        f.write(f"Removed {removed_too_short_text} because the text was shorter than {cfg.run.min_len_tokens}.\n")


def prepare_xsum(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_DATASET = os.path.join(ARTEFACT_PATH, "DATASET.txt")
    ensure_exists_dirs([CACHE_PATH, DATASET_PATH, ARTEFACT_PATH, PREPARED_DATA_PATH])    


    train_set = load_dataset(cfg.dataset.hf_name, cache_dir=CACHE_PATH,
                            trust_remote_code=True)["train"]
    validation_set = load_dataset(cfg.dataset.hf_name, cache_dir=CACHE_PATH,
                                trust_remote_code=True)["validation"]
    test_set = load_dataset(cfg.dataset.hf_name, cache_dir=CACHE_PATH,
                                trust_remote_code=True)["test"]

    dataset = pd.concat([train_set.to_pandas(), validation_set.to_pandas(), test_set.to_pandas()])
    dataset["id"] = dataset["id"].astype(int)
    dataset = dataset.set_index("id")
    dataset.index.name = "doc_id"

    print("Clean dataset")
    dataset[cfg.dataset.text_name] = dataset[cfg.dataset.text_name].map(string_cleaning)
    dataset[cfg.dataset.task_name] = dataset[cfg.dataset.task_name].map(string_cleaning)

    print("Spacy tag the dataset")
    word_counts, texts_shortend = word_count_and_shortend(dataset[cfg.dataset.text_name].astype(str).to_list(), max_words = cfg.run.min_len_tokens, cap = cfg.run.token_cap)
    
    n = len(dataset)
    dataset[cfg.dataset.text_name + "_word_count"] = word_counts
    dataset[cfg.dataset.text_name] = texts_shortend 

    dataset[dataset[cfg.dataset.text_name + "_word_count"] < cfg.run.min_len_tokens].to_csv(os.path.join(ARTEFACT_PATH, cfg.dataset.name + "_story_too_short.csv"))
    dataset = dataset[dataset[cfg.dataset.text_name + "_word_count"] >= cfg.run.min_len_tokens]
    removed_too_short_text = n - len(dataset)
    print("Save dataset")
    dataset.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"))
    
    with open(DOCUMENTATION_FILE_DATASET, "w") as f:
        f.write(f"Removed {removed_too_short_text} because the text was shorter than {cfg.run.min_len_tokens}.\n")


nssec_dict = {
"1": "Higher managerial, administrative and professional occupations",
"1_1": "Large employers and higher managerial and administrative occupations",
"1_2": "Higher professional occupations",
"2": "Lower managerial, administrative and professional occupations",
"3": "Intermediate occupations",
"4": "Small employers and own account workers",
"5": "Lower supervisory and technical occupations",
"6": "Semi-routine occupations",
"7": "Routine occupations",
"8": "Never worked and long-term unemployed",
"*": "Students/unclassifiable",
"uncat": "Unkown",
"unknown": "Unkown"
}

def shorten_conversation_with_tags(row):
    """ Shorten the conversation with tags to the same length as the clean conversations"""
    # Split into a list of conversation turns
    splitted_clean = row["Conversation_Clean"].split("\n")
    splitted_conversation = row["Conversation"].split("\n")

    # Take the same number of conversation turns for the texts with speaker tags
    splitted_conversation = splitted_conversation[:len(splitted_clean)]
    # For the last turn, extract the speaker tag and use the possible shorten version from the clean texts.
    splitted_conversation[-1] = splitted_conversation[-1].split(":")[0] + ": " + splitted_clean[-1] 
    return "\n".join(splitted_conversation)

def prepare_BNC2014Spoken(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_DATASET = os.path.join(ARTEFACT_PATH, "DATASET.txt")
    ensure_exists_dirs([CACHE_PATH, DATASET_PATH, ARTEFACT_PATH, PREPARED_DATA_PATH])
    random.seed(cfg.run.seed)

    
    # Compile regex patterns
    stray_punctuation_pattern = re.compile(r'\s*[^\w\s]+(?=\s|$)')  
    space_before_punct_pattern = re.compile(r'\s+([.,!?])')    
    add_period_pattern = re.compile(r'(?<![.!?])\s*$')       

    # Extract the bnc2014 spoken dataset into the cache
    if os.path.isfile(os.path.join(cfg.datapath, "bnc2014spoken-xml.zip")):
        if not os.path.isdir(os.path.join(CACHE_PATH, "bnc2014spoken-xml")):
            print("Unzipping the BNC 2014 spoken")
            subprocess.run(["unzip", os.path.join(cfg.datapath, "bnc2014spoken-xml.zip"),
                        "-d", os.path.join(CACHE_PATH, "bnc2014spoken-xml")])
    else:
        print(f"Please put the BNC2014-spoken dataset as a zip at: {os.path.join(cfg.datapath, 'bnc2014spoken-xml.zip')}")

    # Get the list of english female and male names from wikipedia
    male_names = get_names("male")
    female_names = get_names("female")
    # Use faker for GB for other pseudomization 
    fake = Faker("en_GB")

    corpus_dict = {
        "id": [],
        "Conversation": [],
        "Conversation_Clean": [],
        "Conversation_Context": [], 
        "Speaker_Metadata": []

    }

    samples_corrupted_metadata = 0
    # Go through all BNC2014 spoken untagged xmls
    for sample_path in glob.glob(os.path.join(CACHE_PATH, "bnc2014spoken-xml", "spoken", "untagged", "*"))[1:]:
        # open the xml tree
        tree = ET.parse(sample_path)
        root = tree.getroot()
        text_id = root.attrib["id"]
        # Get the meta information for the conversation
        header = root.find("header")

        # Keep a dict of the speakers to insert them during the conversation
        speaker_dict = {}
        speaker_dict["females"] = []
        speaker_dict["males"] = []

        # Copy the list of english names and shuffel them to draw random names
        male_names_copy = male_names.copy()
        female_names_copy = female_names.copy()
        random.shuffle(male_names_copy)
        random.shuffle(female_names_copy)

        for key in ["rec_loc", "relationships", "topics", "activity", "conv_type"]:
            if header.find(key).text is None:
                header.find(key).text = "Unkown"

        context_description = "Conversation context:\n" \
        f"  Location: {capitalize_first(header.find("rec_loc").text)}\n"\
        f"  Speaker Relationship: {capitalize_first(header.find("relationships").text)}\n"\
        f"  Conversation Topics: {capitalize_first(header.find("topics").text)}\n"\
        f"  Activity: {capitalize_first(header.find("activity").text)}\n"\
        f"  Conversation Type: {capitalize_first(header.find("conv_type").text)}"\

        speaker_description = "Speaker information:"
        speaker_num = 1
        # Fill up the speaker dict
        for speaker in header.find("speakerInfo"):
            if speaker.find("gender").text == "F":
                speaker_dict[speaker.attrib["id"]] =  f"Speaker_{speaker_num}"
                speaker_num += 1
                speaker_dict["females"].append(speaker_dict[speaker.attrib["id"]])
            elif speaker.find("gender").text == "M":
                speaker_dict[speaker.attrib["id"]] =  f"Speaker_{speaker_num}"
                speaker_num += 1
                speaker_dict["males"].append(speaker_dict[speaker.attrib["id"]])

            for key in ["nat", "lingorig", "birthplace", "birthcountry", "occupation", "edqual"]:
                if speaker.find(key).text is None:
                    speaker.find(key).text = "Unkown"


            speaker_description += f"""
    {speaker_dict[speaker.attrib["id"]]}:
        Age range: {speaker.find("agerange").text.replace("_", "-")}
        Gender: {speaker.find("gender").text}
        Nationality: {capitalize_first(speaker.find("nat").text)}
        Dialect: {capitalize_first(speaker.find("dialect_l1").text)}, {capitalize_first(speaker.find("dialect_l2").text)}, {capitalize_first(speaker.find("dialect_l3").text)}, {capitalize_first(speaker.find("dialect_l4").text)}
        Linguistic Origin: {capitalize_first(speaker.find("lingorig").text)}
        Birthplace: {capitalize_first(speaker.find("birthplace").text)}, {capitalize_first(speaker.find("birthcountry").text)}
        Occupation: {capitalize_first(speaker.find("occupation").text)}
        Highest Level of Education: {capitalize_first(speaker.find("edqual").text[2:])}
        Socio-Economic status: {nssec_dict[speaker.find("nssec").text]}"""

        problematic_files = []
        is_problematic = False
        sentences = []
        # Loop trough the conversation
        for sentence in root.find("body"):
            if is_problematic:
                samples_corrupted_metadata += 1
                break

            # Ignore events, only look at conversatons
            if sentence.tag == "u":
                # Get the speaker of the sentence. If unkown pick randomly.
                if sentence.attrib["who"] == "UNKMALE":
                    if len(speaker_dict["males"]):
                        speaker = random.choice(speaker_dict["males"])
                    else: 
                        speaker = random.choice(speaker_dict["females"])
                elif sentence.attrib["who"] == "UNKFEMALE":
                    if len(speaker_dict["females"]):
                        speaker = random.choice(speaker_dict["females"])
                    else: 
                        speaker = random.choice(speaker_dict["males"])
                elif sentence.attrib["who"] == "UNKMULTI":
                    speaker = random.choice(speaker_dict["females"] + speaker_dict["males"])
                else:
                    # If the speaker ID was not in the meta data, skip this example
                    try:
                        speaker = speaker_dict[sentence.attrib["who"]]
                    except:
                        problematic_files.append(sample_path)
                        is_problematic = True
                        break
                
                # Get the first text of the conversation if available
                if sentence.text:
                    text = sentence.text
                else:
                    text = ""

                # Loop through the sub-tags in a sentence
                for sentence_component in sentence:
                    # If text is unclear, foreign or shift (singing) just keep the text 
                    if sentence_component.tag in ["unclear", "foreign", "shift"]:
                        if sentence_component.text:
                            text += sentence_component.text
                        if sentence_component.tail:
                            text += sentence_component.tail
                    # For vocals do nothing
                    elif sentence_component.tag == "vocal":
                        if sentence_component.tail:
                            text += sentence_component.tail
                    # For anonymized parts
                    elif sentence_component.tag == "anon":
                        # For names, draw a random name based on gender
                        if sentence_component.attrib["type"] == "name":
                            if sentence_component.attrib["nameType"] == "m":
                                text += male_names_copy.pop()
                            elif sentence_component.attrib["nameType"] == "f":
                                text += female_names_copy.pop()
                            elif sentence_component.attrib["nameType"] == "n":
                                if bool(random.getrandbits(1)):
                                    text += male_names_copy.pop()
                                else:
                                    text += female_names_copy.pop()
                            else:
                                raise Exception
                        # For places just take a city
                        elif sentence_component.attrib["type"] == "place":
                            text += fake.city()
                        elif sentence_component.attrib["type"] == "telephoneNumber":
                            text += fake.phone_number()
                        elif sentence_component.attrib["type"] == "address":
                            text += fake.address()
                        elif sentence_component.attrib["type"] == "email":
                            text += fake.ascii_email()
                        elif sentence_component.attrib["type"] == "financialDetails":
                            text += fake.iban()
                        elif sentence_component.attrib["type"] == "socialMediaName":
                            text += fake.user_name()
                        elif sentence_component.attrib["type"] == "dateOfBirth":
                            text += str(fake.date_of_birth())  
                        # Skip personal information, as this varies too much on context.
                        elif sentence_component.attrib["type"] == "miscPersonalInfo":
                            pass
                        else:
                            # Raise an exception, as this should not be possible
                            raise Exception
                        
                        if sentence_component.text:
                            text += sentence_component.text
                        if sentence_component.tail:
                            text += sentence_component.tail
                    # Replace a pause by a comma
                    elif sentence_component.tag == "pause":
                        if text != "":
                            text += ","
                        if sentence_component.tail:
                            text += sentence_component.tail
                    # For trunc and event just use the tail text. Text between trunc often is missing too much.
                    elif sentence_component.tag in  ["trunc", "event"]:
                        if text != "":
                            text += "."
                        if sentence_component.tail:
                            text += sentence_component.tail
                    else:
                        # Raise an exception as the code should not reach this part
                        raise Exception
            
            if is_problematic:
                continue
            else:
                text = capitalize_first(text.strip())
                text = stray_punctuation_pattern.sub('', text)
                text = space_before_punct_pattern.sub(r'\1', text)
                text = add_period_pattern.sub('.', text.strip())
                if text != ".":
                    sentences.append((speaker, text))

            
        conversation = ""
        conversation_clean = ""
        for speaker, text in sentences:
            conversation += f"{speaker}: {text}\n"
            conversation_clean += f"{text}\n"
        conversation = conversation.strip()

        corpus_dict["id"].append(text_id)
        corpus_dict["Conversation"].append(conversation)
        corpus_dict["Conversation_Clean"].append(conversation_clean)
        corpus_dict["Conversation_Context"].append(context_description)
        corpus_dict["Speaker_Metadata"].append(speaker_description)

    dataset = pd.DataFrame(corpus_dict).set_index("id")
    print(dataset)
    print(f"Apply token limit of {cfg.run.min_len_tokens} to the dataset")
    n = len(dataset)

    word_counts, texts_shortend = word_count_and_shortend(dataset["Conversation_Clean"].astype(str).to_list(), max_words = cfg.run.min_len_tokens, cap = cfg.run.token_cap, n_process=1)
    dataset["Conversation_Clean_word_count"] = word_counts
    dataset["Conversation_Clean"] = texts_shortend 
    dataset[dataset["Conversation_Clean_word_count"] < cfg.run.min_len_tokens].to_csv(os.path.join(ARTEFACT_PATH, "BNC2014_Conversation_too_short.csv"))
    dataset = dataset[dataset["Conversation_Clean_word_count"] >= cfg.run.min_len_tokens]
    dataset["Conversation"] = dataset.apply(shorten_conversation_with_tags, axis=1)
    print("Save dataset")
    dataset.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"))
    print(dataset)
        
    with open(DOCUMENTATION_FILE_DATASET, "w") as f:
        f.write(f"Removed {samples_corrupted_metadata} samples because the speaker id did not match with the header.\n")
        f.write(f"Removed {n - len(dataset)} because the conversation was shorter than {cfg.run.min_len_tokens}.\n")


def prepare_writing_prompts(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_DATASET = os.path.join(ARTEFACT_PATH, "DATASET.txt")
    ensure_exists_dirs([CACHE_PATH, DATASET_PATH, ARTEFACT_PATH, PREPARED_DATA_PATH])

    print(f"Load {cfg.dataset.name} dataset")
    train_set = load_dataset("euclaise/writingprompts", cache_dir=CACHE_PATH,
                                        trust_remote_code=True)["train"]
    validation_set = load_dataset("euclaise/writingprompts", cache_dir=CACHE_PATH,
                            trust_remote_code=True)["validation"]
    test_set = load_dataset("euclaise/writingprompts", cache_dir=CACHE_PATH,
                            trust_remote_code=True)["test"]

    # Merge the splitted dataset and reindex it
    dataset = pd.concat([train_set.to_pandas(), validation_set.to_pandas(), test_set.to_pandas()], ignore_index=True)
    dataset.index.name = "doc_id"
    print("Clean dataset")
    # Filter out prompts shorter than the WP Tag
    #dataset = dataset[dataset["prompt"].map(lambda x: len(x)) > len("[ WP ]")]
    n = len(dataset)
    # Remove all prompts without the WP Tag
    dataset = dataset[dataset["prompt"].map(lambda x: x[0:6] == "[ WP ]")]
    removed_wrong_tag = n - len(dataset)

    # Apply string cleaning method. For the prompt also remove the [ WP ] tag
    dataset["prompt"] = dataset["prompt"].map(lambda x: string_cleaning(x[7:]))
    dataset["story"] = dataset["story"].map(string_cleaning)

    # Remove all stories shorter than 100 words (required by subreddit rules)
    dataset = dataset[dataset["story"].map(lambda x: len(x.split()) > 99 )]

    # Write out prompts sorted by word length for manual checks
    dataset.sort_values(by='prompt', key=lambda col: col.str.split().str.len()).iloc[range(1000)
                    ].to_csv(os.path.join(ARTEFACT_PATH, 'WritingPrompts_Promptlength.txt'), index=False, header=False)
    
    n = len(dataset)
    # Remove all prompts shorter than a threshold defined manually by the previous artefact
    dataset[dataset['prompt'].str.split().str.len() <= cfg.dataset.min_word_count_prompt].to_csv(os.path.join(ARTEFACT_PATH, "WritingPrompts_prompt_too_short.csv"))
    dataset = dataset[dataset['prompt'].str.split().str.len() > cfg.dataset.min_word_count_prompt]
    removed_too_short_prompt = n - len(dataset)

    dataset['punct_ratio_prompt'] = dataset['prompt'].apply(lambda x: punctuation_ratio(str(x)))
    dataset['punct_ratio_story'] = dataset['story'].apply(lambda x: punctuation_ratio(str(x)))

    dataset[dataset["punct_ratio_prompt"] > cfg.dataset.max_punctuation_ratio]["prompt"].to_csv(os.path.join(ARTEFACT_PATH, "WritingPrompts_prompt_punct.csv"))
    dataset[dataset["punct_ratio_story"] > cfg.dataset.max_punctuation_ratio]["story"].to_csv(os.path.join(ARTEFACT_PATH, "WritingPrompts_story_punct.csv"))
    
    n = len(dataset)
    dataset = dataset[dataset["punct_ratio_story"] <= cfg.dataset.max_punctuation_ratio]
    removed_corrupted = n - len(dataset)

    #print("Plot dataset statistics")
    #plot_length_hist(dataset, "story", os.path.join(ARTEFACT_PATH, "WritingPrompts_story_length_hist.png"))
    #plot_length_hist(dataset, "prompt", os.path.join(ARTEFACT_PATH, "WritingPrompts_prompt_length_hist.png"))


    print("Get docs of texts from spacy")
    #dataset["docs_story"] = get_parsed_text(dataset["story"].astype(str).tolist())

    print(f"Apply token limit of {cfg.run.min_len_tokens} to the dataset")
    n = len(dataset)

    word_counts, texts_shortend = word_count_and_shortend(dataset["story"].astype(str).to_list(), max_words = 400, cap = 440)
    dataset["story_word_count"] = word_counts
    dataset["story"] = texts_shortend 

    dataset[dataset["story_word_count"] < cfg.run.min_len_tokens].to_csv(os.path.join(ARTEFACT_PATH, "WritingPrompts_story_too_short.csv"))
    dataset = dataset[dataset["story_word_count"] >= cfg.run.min_len_tokens]

    


    removed_too_short_text = n - len(dataset)

    print("Save dataset")
    dataset.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"))
    
    with open(DOCUMENTATION_FILE_DATASET, "w") as f:
        f.write(f"Removed {removed_wrong_tag} samples because they had the wrong tag.\n")
        f.write(f"Removed {removed_too_short_prompt} samples because the prompt was shorter than {cfg.dataset.min_word_count_prompt + 1}.\n")
        f.write(f"Removed {removed_corrupted} samples because their token to punctuation ratio was too high.\n")
        f.write(f"Removed {removed_too_short_text} because the text was shorter than {cfg.run.min_len_tokens}.\n")



@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")

    print(f"Dataset: {cfg.dataset.name}")
    if cfg.dataset.name == "WritingPrompts":
        prepare_writing_prompts(cfg)
    elif cfg.dataset.name == "XSum":
        prepare_xsum(cfg)
    elif cfg.dataset.name == "BNC2014Spoken":
        prepare_BNC2014Spoken(cfg)
    elif cfg.dataset.name == "wikiHow":
        prepare_wikiHow(cfg)
    elif cfg.dataset.name == "S2ORC_ACL":
        prepare_s2orc_ACL(cfg)
    else:
        print(f"{cfg.dataset.name} is not a valid dataset.")



if __name__ == "__main__":
    main()



