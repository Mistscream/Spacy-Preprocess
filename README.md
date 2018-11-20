## Preprocessing

**Install:**
The project uses pipenv to manage dependencies. You can install all requirements with the following command:

    $ pipenv install
    $ pipenv shell
    $ pipenv run python -m spacy download de
    $ pipenv run python -m ipykernel install --user --name mygreatenv --display-name "My Great Env"


**Run jupyter notebook:**

    $ pipenv run jupyter notebook
    
**Still ToDo:**

 - edit stopword list
 - edit Tag list
 - better tokenizing for things like "28-jähriger" 
 - maybe extend custom lemmatization json file (much work, for less output?)