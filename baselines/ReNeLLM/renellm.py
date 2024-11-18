import random

from utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from utils.scenario_nest_utils import SCENARIOS
from utils.harmful_classification_utils import harmful_classification


def rene(instruct, rene_args):
    temp_harm_behavior = instruct
    
    operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
    scenarios = SCENARIOS
    print(
    "\n################################\n"
    f"Current Data: {instruct}\n"
    "################################\n")
    # prompt rewriting
    while True:
        print(f"******* Start Prompt Rewriting! *******")
        # randomly select the number of rewriting operations.
        n = random.randint(1, 6)
        # arrange the execution order randomly
        operation_indexes = random.sample(range(6), n)
        print(f"The number of rewriting functions is: {n}")
        print(f"The order of the rewriting operations is: {operation_indexes}\n")

        temp_rewrite_results = [['original prompt', temp_harm_behavior]]
        for index in operation_indexes:
            print(f"Excute function {index}: {operations[index].__name__}")
            instruct = operations[index](rene_args, instruct)
            print(f"Rewritten prompt: {instruct}\n")
            temp_rewrite_results.append([operations[index].__name__, instruct])
            
        # check if the prompt meets the requirements; if it does, break out of the loop.
        print("******* Check if the prompt has been modified to be harmless *******")
        label = harmful_classification(rene_args, instruct)
        if label == "1":
            print(f"******* Prompt Rewriting Done! *******\n")
            break
        else:
            print(f"******* Prompt Rewriting Failed! *******\n")
            instruct = temp_harm_behavior
            continue

    # scenario nesting
    print(f"******* Start Prompt Scenario Nesting! *******")
    print(f"harm_behavior:\n{instruct}\n")
    # randomly select a scenario
    scenario = random.choice(scenarios)
    nested_prompt = scenario.replace("<>", instruct)
    print(f"nested_prompt:\n{nested_prompt}\n")
    
    return(nested_prompt, temp_harm_behavior)