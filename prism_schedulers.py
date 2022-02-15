from collections import defaultdict
from pathlib import Path

import aalpy.paths
from aalpy.utils import mdp_2_prism_format


class PrismInterface:
    def __init__(self, dest, model, num_steps=None):
        self.tmp_dir = Path("tmp_prism")
        self.dest = dest
        self.model = model
        self.num_steps = num_steps
        self.tmp_mdp_file = (self.tmp_dir / f"po_rl_{dest}.prism")
        # self.tmp_prop_file = f"{self.tmp_dir_name}/po_rl.props"
        self.current_state = None
        self.tmp_dir.mkdir(exist_ok=True)
        self.prism_property = self.create_mc_query()
        mdp_2_prism_format(self.model, "porl", output_path=self.tmp_mdp_file)
        self.adv_file_name = (self.tmp_dir.absolute() / f"sched_{dest}.adv")
        self.concrete_model_name = str(self.tmp_dir.absolute() / f"concrete_model_{dest}")
        self.property_val = None
        self.call_prism()
        self.parser = PrismSchedulerParser(self.adv_file_name, self.concrete_model_name + ".lab",
                                           self.concrete_model_name + ".tra")

    def create_mc_query(self):
        prop = f"Pmax=?[F \"{self.dest}\"]" if not self.num_steps else f'Pmax=?[F<{self.num_steps} \"{self.dest}\"]'
        return prop

    def get_input(self):
        if self.current_state is None:
            # print("Return none because current state is none")
            return None
        else:
            # print("Current state is not none")
            if self.current_state not in (self).parser.scheduler_dict:
                return None
            return self.parser.scheduler_dict[self.current_state]

    def reset(self):
        self.current_state = self.parser.initial_state

    def step_to(self, input, output):
        trans_from_current = self.parser.transition_dict[self.current_state]
        found_state = False
        for (prob, action, target_state) in trans_from_current:
            if action == input and output in self.parser.label_dict[target_state]:
                self.current_state = target_state
                found_state = True
                break
        if not found_state:
            self.current_state = None

        return found_state

    def call_prism(self):
        import subprocess
        import io
        from os import path

        prism_file = aalpy.paths.path_to_prism.split('/')[-1]
        path_to_prism_file = aalpy.paths.path_to_prism[:-len(prism_file)]
        file_abs_path = path.abspath(self.tmp_mdp_file)
        results = []
        proc = subprocess.Popen(
            [aalpy.paths.path_to_prism, file_abs_path, "-pf", self.prism_property, "-noprob1", "-exportadvmdp",
             self.adv_file_name, "-exportmodel", f"{self.concrete_model_name}.all"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=path_to_prism_file, shell=True)
        out = proc.communicate()[0]
        out = out.decode('utf-8').splitlines()
        for line in out:
            if not line:
                continue
            if 'Syntax error' in line:
                print(line)
            else:
                if "Result:" in line:
                    end_index = len(line) if "+/-" not in line else line.index("(") - 1
                    try:
                        result_val = float(line[len("Result: "): end_index])
                        # if result_val < 1.0:
                        #    print(f"We cannot reach with absolute certainty, probability is {result_val}")
                        results.append(result_val)
                    except:
                        print("Result parsing error")
        proc.kill()
        if len(results) == 1:
            self.property_val = results[0]
        return results


class PrismSchedulerParser:
    def __init__(self, scheduler_file, label_file, transition_file):
        with open(scheduler_file, "r") as f:
            self.scheduler_file_content = f.readlines()
        with open(label_file, "r") as f:
            self.label_file_content = f.readlines()
        with open(transition_file, "r") as f:
            self.transition_file_content = f.readlines()
        self.label_dict = self.create_labels()
        self.transition_dict = self.create_transitions()
        self.scheduler_dict = self.parse_scheduler()
        self.initial_state = next(filter(lambda e: "init" in e[1], self.label_dict.items()))[0]
        self.actions = set()
        for l in self.transition_dict.values():
            for _, action, _ in l:
                self.actions.add(action)
        self.actions = list(self.actions)

    def create_labels(self):
        label_dict = dict()
        header_line = self.label_file_content[0]
        label_lines = self.label_file_content[1:]
        header_dict = dict()
        split_header = header_line.split(" ")
        for s in split_header:
            label_id = s.strip().split("=")[0]
            label_name = s.strip().split("=")[1].replace('"', '')
            header_dict[label_id] = label_name
        for l in label_lines:
            state_id = int(l.split(":")[0])
            label_ids = l.split(":")[1].split(" ")
            label_names = set(
                map(lambda l_id: header_dict[l_id.strip()], filter(lambda l_id: l_id.strip(), label_ids)))
            label_dict[state_id] = label_names
        return label_dict

    def create_transitions(self):
        header_line = self.transition_file_content[0]
        transition_lines = self.transition_file_content[1:]
        transitions = defaultdict(list)
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            target_state = int(split_line[2])
            prob = float(split_line[3])
            action = split_line[4].strip()
            transitions[source_state].append((prob, action, target_state))
        return transitions

    def parse_scheduler(self):
        header_line = self.scheduler_file_content[0]
        transition_lines = self.scheduler_file_content[1:]
        scheduler = dict()
        for t in transition_lines:
            split_line = t.split(" ")
            source_state = int(split_line[0])
            action = split_line[4].strip()
            if source_state in scheduler:
                assert action == scheduler[source_state]
            else:
                scheduler[source_state] = action
        return scheduler
