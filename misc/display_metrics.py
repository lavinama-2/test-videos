import os, json
import pandas as pd
TRAINING_FILES = True

path_to_json = "scripts/out/MultiAgentAdvIntersectionEnv/DQNAgent/one_npc/baseline"
last_bit_string = "_20220827-171240_620943"
example_model_name = "three_npc_failmaker_ego_attention_20220828-123241_659178"
files = []
metrics = {}
path_type_agent = "scripts/out/MultiAgentAdvIntersectionEnv/DQNAgent/one_npc"
path_number_agents = "scripts/out/MultiAgentAdvIntersectionEnv/DQNAgent"
path_algorithm = "scripts/out/MultiAgentAdvIntersectionEnv"

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def calc_percentages(raw_metrics):
    """
    Calculate the percentage of crashes given the metrics file
    :param metrics_file: dict loaded from json files
    :return: percentages of crashes (dict) 
    """
    crashes = raw_metrics["crashes"]
    percentages = {}
    for vehicle, num in crashes.items():
        if TRAINING_FILES:
            percentages[vehicle] = round(num/10000 * 100, 2)
        else:
            percentages[vehicle] = round(num/1000 * 100, 2)
    return percentages

def print_lines(algorithm,number_agents,type_agent,name_trained_agent,train_type,vehicle,per):
    string_to_print = "{:<10} {:<10} {:<15} {:<12} {:<6} {:<8} {:<4}".format(algorithm,number_agents,type_agent,name_trained_agent,train_type,vehicle,per)
    if vehicle == "ego":
        string_to_print = color.BOLD + string_to_print + color.END
    if "zero_sum" in name_trained_agent:
        string_to_print = color.PURPLE + string_to_print + color.END
    if "rule_break" in name_trained_agent:
        string_to_print = color.CYAN + string_to_print + color.END
    if "failmaker" in name_trained_agent:
        string_to_print = color.GREEN + string_to_print + color.END
    if "good_agent" in name_trained_agent:
        string_to_print = color.YELLOW + string_to_print + color.END
    if "background" in name_trained_agent:
        string_to_print = color.RED + string_to_print + color.END
    if "test" in train_type:
        string_to_print = color.UNDERLINE + string_to_print + color.END
    print(string_to_print)


algorithms = []
list_num_agents = []
types_agents = []
final_dict = {}
train_type = ""
print("{:<10} {:<10} {:<15} {:<12} {:<6} {:<8} {:<4}".format('algorithm','number_agents','type_agent','Agent','Train Type','Vehicle','%'))

# Look for: DQNAgent, MADDPGAgent
for algorithm in os.listdir(path_algorithm):
    if algorithm == "DQNAgent" or algorithm == "MADDPGAgent":
        path_number_agents = os.path.join(path_algorithm, algorithm)
        metrics[algorithm] = {}
        algorithms.append(algorithm)

        # Look for: one_npc, three_npc
        for number_agents in os.listdir(path_number_agents):
            if number_agents == "one_npc" or number_agents == "three_npc":
                path_type_agent = os.path.join(path_number_agents, number_agents)
                metrics[algorithm][number_agents] = {}
                list_num_agents.append(number_agents)

                # Look for: baseline, ego_attention
                for type_agent in os.listdir(path_type_agent):
                    if type_agent == "baseline" or type_agent == "ego_attention":
                        path_to_json = os.path.join(path_type_agent, type_agent)
                        metrics[algorithm][number_agents][type_agent] = {}
                        types_agents.append(type_agent)
                        print("---------------------------------------------------------------------------")
                        # Look for: zero_sum, failmaker, background, ...
                        for trained_agents in os.listdir(path_to_json):
                            if trained_agents[0] != ".":
                                path_trained_agents = os.path.join(path_to_json, trained_agents)

                                for json_file_name in os.listdir(path_trained_agents):
                                    # print("i", i)
                                    if os.path.isfile(os.path.join(path_trained_agents, json_file_name)) and 'metrics' in json_file_name:
                                        path_json = os.path.join(path_trained_agents, json_file_name)
                                        files.append(json_file_name)
                                        with open(path_json) as json_file:
                                            name_trained_agent = trained_agents[len(number_agents)+1:-(len(last_bit_string)+len(type_agent)+1)]
                                            if "train" in name_trained_agent:
                                                name_trained_agent = name_trained_agent.replace("_train", "")
                                                train_type = "train"
                                            if "test" in name_trained_agent:
                                                name_trained_agent = name_trained_agent.replace("_test", "")
                                                train_type = "test"
                                            
                                            metrics[algorithm][number_agents][type_agent][name_trained_agent] = {}
                                            metrics_file = json.load(json_file)
                                            percentages = calc_percentages(metrics_file)
                                            for vehicle, per in percentages.items():
                                                metrics[algorithm][number_agents][type_agent][name_trained_agent][vehicle] = per
                                                final_dict[name_trained_agent + "_" + vehicle] = [algorithm, number_agents, type_agent, per]
                                                print_lines(algorithm,number_agents,type_agent,name_trained_agent,train_type,vehicle,per)
                        
"""
print("{:<30} {:<15} {:<10} {:<12} {:<8}".format('Agent + Vehicle','algorithm','number_agents','type_agent','per'))
for name_trained_agent_vehicle, v in final_dict.items():
    algorithm, number_agents, type_agent, name_trained_agent = v
    print("{:<30} {:<15} {:<10} {:<12} {:<8}".format(name_trained_agent_vehicle, algorithm, number_agents, type_agent, name_trained_agent))
"""



"""
for file_name, metrics_file in metrics.items():
    crashes = metrics_file["crashes"]
    percentage = {}
    for vehicle, num in crashes.items():
        if TRAINING_FILES:
            percentage[vehicle] = round(num/10000 * 100, 2)
        else:
            percentage[vehicle] = round(num/1000 * 100, 2)
    print(f"{file_name}: {percentage}")
"""
