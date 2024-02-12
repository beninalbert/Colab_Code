import subprocess

# Install python packages
subprocess.run(['pip', 'install', 'torch', 'transformers', 'bitsandbytes', 'accelerate', 'sqlparse'])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.is_available()

available_memory = torch.cuda.get_device_properties(0).total_memory
print(available_memory)

model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if available_memory > 15e9:
    # if you have atleast 15GB of GPU memory, run load the model in float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
else:
    # else, load in 8 bits – this is a bit slower
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        use_cache=True,
    )

test_prompt = """
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'
- Remember that revenue is price multiplied by quantity
- Remember that cost is supply_price multiplied by quantity

### Database Schema
This query will run on a database whose schema is represented in this string:




CREATE TABLE airfields (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
IdentityType VARCHAR(30),
Category VARCHAR(255),
EntityName VARCHAR(255),
LocationDetails VARCHAR(255),
GR VARCHAR(255),
DateTime DATE(8),
Bubble DECIMAL(8),
RightLabel VARCHAR(50),
LeftLabel VARCHAR(50),
Range_ DECIMAL(8),
AddlInfo VARCHAR(255),
Remarks VARCHAR(255),
Id INTEGER(4),
Identifier VARCHAR(100),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
Created_date DATE(8),
Last_edited_date DATE(8),
created_user VARCHAR(255),
last_edited_user VARCHAR(255),
CoverageAngle DECIMAL(8),
Field1 INTEGER(4),
Field2 INTEGER(4),
Field3 INTEGER(4),
Field4 VARCHAR(255),
Field5 VARCHAR(255),
base_name VARCHAR(100),
rw_length DECIMAL(8),
ptt_avlb VARCHAR(5),
no_of_fighter_sqns INTEGER(4),
no_of_tpt_sqns INTEGER(4),
no_of_heptr_sqns INTEGER(4),
sagw_unit VARCHAR(5));




CREATE TABLE otrfacilities (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
IdentityType VARCHAR(30),
Category VARCHAR(255),
EntityName VARCHAR(255),
LocationDetails VARCHAR(255),
GR VARCHAR(255),
DateTime DATE(8),
Bubble DECIMAL(8),
RightLabel VARCHAR(50),
LeftLabel VARCHAR(50),
Range_ DECIMAL(8),
AddlInfo VARCHAR(255),
Remarks VARCHAR(255),
Id INTEGER(4),
Identifier VARCHAR(100),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
Created_date DATE(8),
Last_edited_date DATE(8),
created_user VARCHAR(255),
last_edited_user VARCHAR(255),
CoverageAngle DECIMAL(8),
Field1 INTEGER(4),
Field2 INTEGER(4),
Field3 INTEGER(4),
Field4 VARCHAR(255),
Field5 VARCHAR(255),
berth_length INTEGER(4),
pol_availability VARCHAR(5),
port_limits VARCHAR(100));




CREATE TABLE Own_Deployment (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
remarks VARCHAR(100),
gr VARCHAR(20),
identifier VARCHAR(50),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
leftlabel VARCHAR(255),
rightlabel VARCHAR(255),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
bubble DECIMAL(8));




CREATE TABLE Enemy_Deployment (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
remarks VARCHAR(100),
gr VARCHAR(20),
height DECIMAL(8),
identifier VARCHAR(50),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
leftlabel VARCHAR(255),
rightlabel VARCHAR(255),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
bubble DECIMAL(8));




CREATE TABLE Radars (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
remarks VARCHAR(100),
gr VARCHAR(20),
range_ DECIMAL(8),
coverageangle DECIMAL(8),
height DECIMAL(8),
identifier VARCHAR(50),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
verticalangle DECIMAL(8),
lookangle DECIMAL(8),
groundheight DECIMAL(8),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
bubble DECIMAL(8));




CREATE TABLE WeaponsAndSensors (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
remarks VARCHAR(100),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
gr VARCHAR(20),
range_ DECIMAL(8),
coverageangle DECIMAL(8),
height DECIMAL(8),
identifier VARCHAR(50),
symbolcolor VARCHAR(30),
symbolsize INTEGER(2),
symbolangle DECIMAL(8),
verticalangle DECIMAL(8),
lookangle DECIMAL(8),
groundheight DECIMAL(8),
id INTEGER(4),
leftlabel VARCHAR(100),
rightlabel VARCHAR(100),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
bubble DECIMAL(8));




CREATE TABLE OwnDeploymentArea (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
name VARCHAR(100),
gr VARCHAR(20),
identifier VARCHAR(50),
symbolcolor VARCHAR(30),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
st_area(Shape) DECIMAL(0) PRIMARY KEY,
st_perimeter(Shape) DECIMAL(0) PRIMARY KEY);




CREATE TABLE EnemyDeploymentArea (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
gr VARCHAR(255),
name VARCHAR(255),
identifier VARCHAR(255),
symbolcolor VARCHAR(255),
remarks VARCHAR(255),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
st_area(Shape) DECIMAL(0) PRIMARY KEY,
st_perimeter(Shape) DECIMAL(0) PRIMARY KEY);




CREATE TABLE Mine_Field (
OBJECTID INTEGER(4) PRIMARY KEY,
Shape VARCHAR(8) PRIMARY KEY,
remarks VARCHAR(255),
created_user VARCHAR(255),
created_date DATE(8),
last_edited_user VARCHAR(255),
last_edited_date DATE(8),
gr VARCHAR(50),
identifier VARCHAR(255),
symbolcolor VARCHAR(30),
id INTEGER(4),
identitytype VARCHAR(30),
category VARCHAR(255),
entityname VARCHAR(255),
locationdetails VARCHAR(255),
datetime DATE(8),
st_area(Shape) DECIMAL(0) PRIMARY KEY,
st_perimeter(Shape) DECIMAL(0) PRIMARY KEY);


"""

import sqlparse

def generate_query(question):
    updated_prompt = test_prompt.format(question=question)
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # empty cache so that you do generate more results w/o memory crashing
    # particularly important on Colab – memory management is much more straightforward
    # when running on an inference service
    return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=False)

question = "Get me the airfields in the Jaipur "
generated_sql = generate_query(question)
print(generated_sql)