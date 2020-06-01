SUBMISSION = True #Set this to true for submission. Set it to False if testing on your machine.

import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, Point
import os

################### DO NOT CHANGE THIS ###################
if not SUBMISSION :
    FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = "/fast_downward/"
else :
    FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = ""
PDDL_FILE_ABSOLUTE_PATH = ""
##########################################################
test_case_number = 0
# Sample test cases for Parking task. 
#test_config = [{'lanes' : [LaneSpec(2, [0, 0])] *3,'width' :5, 'seed' : 13},
 #              {'lanes' : [LaneSpec(2, [0, 0])] *3,'width' :5, 'seed' : 10},
  #             {'lanes' : [LaneSpec(3, [0, 0])] *4,'width' :10, 'seed' : 25},
   #            {'lanes' : [LaneSpec(4, [0, 0])] *4,'width' :10, 'seed' : 25},
    #           {'lanes' : [LaneSpec(8, [0, 0])] *7,'width' :20, 'seed' : 25},
     #          {'lanes' : [LaneSpec(7, [0, 0])] *10,'width' :20, 'seed' : 125}]

# ## Sample test cases for Moving cars. 
#test_config = [{'lanes' : [LaneSpec(3, [-1, -1])] *3,'width' :5, 'seed': 12},
 #              {'lanes' : [LaneSpec(2, [-1, -1])] *3,'width' :5, 'seed': 25},
  #             {'lanes' : [LaneSpec(3, [-1, -1])] *4,'width' :10, 'seed': 125},
   #            {'lanes' : [LaneSpec(4, [-1, -1])] *4,'width' :10, 'seed': 28},
    #           {'lanes' : [LaneSpec(8, [-1, -1])] *7,'width' :20, 'seed': 119},
     #          {'lanes' : [LaneSpec(7, [-1, -1])] *10,'width' :20, 'seed': 21}]

## Sample test cases for Crossing Task
test_config = [{'lanes' : [LaneSpec(6, [-2, -2])] *2 + [LaneSpec(6, [-5, -5])] *2 +
                          [LaneSpec(5, [-4, -4])] *2 + [LaneSpec(5, [-2, -2])] *1, 'width' :30, 'seed' : 101}]

# test_config = [{'lanes' : [LaneSpec(6, [-2, -2])] *2 + [LaneSpec(4, [-3, -3])] *2 + [LaneSpec(6, [-4, -4])] *2 + [LaneSpec(4, [-4, -4])] *2 +  [LaneSpec(6, [-2, -2])] *2 + [LaneSpec(4, [-3, -3])] *2 ,  'width' :35, 'seed' : 101}]

# test_case_number = 0 # Change the index for a different test case
LANES = test_config[test_case_number]['lanes']
WIDTH = test_config[test_case_number]['width']
RANDOM_SEED = test_config[test_case_number]['seed']


class GeneratePDDL_Stationary :
    '''
    Class to generate the PDDL files given the environment description.
    '''
    def __init__ (self, env, num_lanes, width, file_name) :
        self.state = env.reset()
        self.num_lanes = num_lanes
        self.width = width
        self.file_name = file_name
        self.problem_file_name = self.file_name + 'problem.pddl' 
        self.domain_file_name = self.file_name + 'domain.pddl' 
        self.domain_string = ""
        self.type_string = ""
        self.predicate_strings = self.addHeader("predicates")
        self.action_strings = ""
        self.problem_string = ""
        self.object_strings = self.addHeader("objects")


    def addDomainHeader(self, name='default_header') :
        '''
        Adds the header in the domain file.

        Parameters : 
        name (string): domain name.
        '''
        self.domain_header = "(define (domain " + name +" ) \n" +"(:requirements :strips :typing) \n"


    def addTypes(self, types={}) :
        '''
        Adds the object types to the PDDL domain file.

        Parameters : 
        types (dict): contains a dictionary of (k,v) pairs, where k is the object type, and v is the supertype. If k has no supertype, v is None.
        '''
        type_string = "(:types "

        for _type, _supertype in types.items() :
            if _supertype is None :
                type_string += _type +  "\n"
            else : 
                type_string += _type + " - " + _supertype + "\n"
        type_string += ") \n"
        self.type_string = type_string


    def addPredicate(self, name='default_predicate', parameters = (), isLastPredicate=False) :
        '''
        Adds predicates to the PDDL domain file

        Parameters : 
        name (string) : name of the predicate.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        isLastPredicate (bool) : True for the last predicate added.
        '''
        predicate_string = "(" + name
        for var_name, var_type in parameters :
            predicate_string += " ?" + var_name + " - " + var_type
        predicate_string += ") \n"
        self.predicate_strings+= predicate_string

        if isLastPredicate :
            self.predicate_strings += self.addFooter()


    def addAction(self, name='default_action', parameters=(), precondition_string= "", effect_string= "") :
        '''
        Adds actions to the PDDL domain file

        Parameters : 
        name (string) : name of the action.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        precondition_string (string) : The precondition for the action.
        effect_string (string) : The effect of the action.
        '''
        action_string = name + "\n"
        parameter_string = ":parameters ("
        for var_name, var_type in parameters :
            parameter_string += " ?" + var_name + " - " + var_type
        parameter_string += ") \n"
        
        precondition_string = ":precondition " + precondition_string + "\n"
        effect_string = ":effect " + effect_string + "\n"
        action_string += parameter_string + precondition_string + effect_string
        action_string = self.addHeader("action") + action_string + self.addFooter()
        self.action_strings+= action_string

    def generateDomainPDDL(self) :
        '''
        Generates the PDDL domain file after all actions, predicates and types are added
        '''
        domain_file = open(PDDL_FILE_ABSOLUTE_PATH + self.domain_file_name, "w")
        PDDL_String = self.domain_header + self.type_string + self.predicate_strings + self.action_strings + self.addFooter()
        domain_file.write(PDDL_String)
        domain_file.close()

   
    def addProblemHeader(self, problem_name='default_problem_name', domain_name='default_domain_name') :
        '''
        Adds the header in the problem file.

        Parameters : 
        problem_name (string): problem name.
        domain_name (string): domain name.
        '''
        self.problem_header = "(define (problem " + problem_name + ") \n (:domain " + domain_name + ") \n"
    

    def addObjects(self, obj_type, obj_list=[], isLastObject=False) :
        '''
        Adds object instances of the same type to the problem file

        Parameters :
        obj_type (string) : the object type of the instances that are being added
        obj_list (list(str)) : a list of object instances to be added
        isLastObject (bool) : True for the last set of objects added.
        '''
        obj_string = ""
        for obj in obj_list :
            obj_string += obj + " "
        obj_string += " - " + obj_type
        self.object_strings += obj_string + "\n "
        if isLastObject :
            self.object_strings += self.addFooter()


    def addInitState(self) :
        '''
        Generates the complete init state
        '''
        initString = self.generateInitString()
        self.initString = self.addHeader("init") + initString + self.addFooter()


    def addGoalState(self) :
        '''
        Generates the complete goal state
        '''
        goalString = self.generateGoalString()
        self.goalString = self.addHeader("goal") + goalString + self.addFooter()


    def generateGridCells(self) :
        '''
        Generates the grid cell objects. 
        
        For a |X| x |Y| sized grid, |X| x |Y| objects to represent each grid cell are created. 
        pt0pt0, pt1pt0, .... ptxpt0
        pt0pt1, pt1pt1, .... ptxpt1
        ..       ..            ..
        ..       ..            ..
        pt0pty, pt1pty, .... ptxpty


        '''
        self.grid_cell_list = []
        for w in range(self.width) :
            for lane in range(self.num_lanes) :
                self.grid_cell_list.append("pt{}pt{}".format(w, lane))
 

    def generateInitString(self) :
        
        # Agent
        initString = "(at pt{}pt{} agent1)".format(self.state.agent.position.x,self.state.agent.position.y)
        initString += "(cur_time t0)"
        
        for t in range (self.width+self.num_lanes):
            initString += "(upd_time t{} t{})".format(t,t+1)
        
        
        # Car position and blocked pts
        for t in range (self.width+self.num_lanes):
            for car in self.state.cars:
                x = car.position.x + car.speed_range[0]*(t+1)
                y = car.position.y
                while x<0:
                    x += self.width
                
                if car.speed_range[0] == 0:
                    initString += "(blocked pt{}pt{} t{})".format(x,y,t)
                else:
                    for trail in range(-car.speed_range[0]):
                        x_trail = x+trail
                        if x_trail >= self.width:
                            x_trail-= self.width
                        initString += "(blocked pt{}pt{} t{})".format(x_trail,y,t)
                   
        # Possible movements
        for x1 in range(self.width):
            for y1 in range(self.num_lanes):
                x2 = x1-1
#                 if x1==0:
#                     x2+=self.width
                
                if x1>0:
                    # up
                    y2 = y1-1
                    if y2>=0:
                        initString+="(up_next pt{}pt{} pt{}pt{})".format(x1,y1,x2,y2)

                    # down
                    y2 = y1+1
                    if y2<self.num_lanes:
                        initString+="(down_next pt{}pt{} pt{}pt{})".format(x1,y1,x2,y2)

                    # forward
                    y2 = y1
                    initString+="(forward_next pt{}pt{} pt{}pt{})".format(x1,y1,x2,y2)  
                
        return initString

    def generateGoalString(self) :
        
        goalString = "(at pt{}pt{} agent1)".format(self.state.finish_position.x,self.state.finish_position.y)
#         goalString += "(blocked pt{}pt{})".format(self.state.finish_position.x,self.state.finish_position.y)
    
        return goalString

    def generateProblemPDDL(self) :
        '''
        Generates the PDDL problem file after the object instances, init state and goal state are added
        '''
        problem_file = open(PDDL_FILE_ABSOLUTE_PATH + self.problem_file_name, "w")
        PDDL_String = self.problem_header + self.object_strings + self.initString + self.goalString + self.addFooter()
        problem_file.write(PDDL_String)
        problem_file.close()


    '''
    Helper Functions 
    '''
    def addHeader(self, name) :
        return "(:" + name + " "

    def addFooter(self) :
        return ") \n"      

def initializeSystem(env):
    gen = GeneratePDDL_Stationary(env, len(env.lanes), width=env.width, file_name='HW1')
    return gen


def generateDomainPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File. 
    As a part of the assignemnt, you will need to add the actions here.
    '''
    gen.addDomainHeader("grid_world")
    gen.addTypes(types = {"car" : None, "agent" : "car", "gridcell" : None, "time" : None})
    '''
    Predicate Definitions :
    at(pt, car) : car is at gridcell pt.
    up_next(pt1, pt2) : pt2 is the next location of the car when it takes the UP action from pt1
    down_next(pt1, pt2) : pt2 is the next location of the car when it takes the DOWN action from pt1
    forward_next(pt1, pt2) : pt2 is the next location of the car when it takes the FORWARD action from pt1
    blocked(pt) : The gridcell pt is occupied by a car and is "blocked".
    '''
    gen.addPredicate(name="upd_time", parameters=(("t1" , "time"), ("t2", "time")))
    gen.addPredicate(name="cur_time", parameters=[("t1" , "time")])
    gen.addPredicate(name="at", parameters=(("pt1" , "gridcell"), ("car", "car")))
    gen.addPredicate(name="up_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="down_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="forward_next", parameters=(("pt1" , "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="blocked", parameters=(("pt1" , "gridcell"),("t1" , "time")) , isLastPredicate=True)
    
    # Actions
    gen.addAction(name="UP", 
                  parameters = (("agent1", "agent"),("pt1" , "gridcell"),("pt2" , "gridcell"),("t1" , "time"), ("t2", "time")), 
                  precondition_string = "(and (up_next ?pt1 ?pt2)(at ?pt1 agent1)(not (blocked ?pt2 ?t1))(cur_time ?t1)(upd_time ?t1 ?t2))",
                  effect_string = "(and (at ?pt2 agent1)(not (at ?pt1 agent1))(not (cur_time ?t1))(cur_time ?t2))")
    gen.addAction(name="DOWN", 
                  parameters = (("agent1", "agent"),("pt1" , "gridcell"),("pt2" , "gridcell"),("t1" , "time"), ("t2", "time")), 
                  precondition_string = "(and (down_next ?pt1 ?pt2)(at ?pt1 agent1)(not (blocked ?pt2 ?t1))(cur_time ?t1)(upd_time ?t1 ?t2))",
                  effect_string = "(and (at ?pt2 agent1)(not (at ?pt1 agent1))(not (cur_time ?t1))(cur_time ?t2))")
    gen.addAction(name="FORWARD[-1]", 
                  parameters = (("agent1", "agent"),("pt1" , "gridcell"),("pt2" , "gridcell"),("t1" , "time"), ("t2", "time")), 
                  precondition_string = "(and (forward_next ?pt1 ?pt2)(at ?pt1 agent1)(not (blocked ?pt2 ?t1))(cur_time ?t1)(upd_time ?t1 ?t2))",
                  effect_string = "(and (at ?pt2 agent1)(not (at ?pt1 agent1))(not (cur_time ?t1))(cur_time ?t2))")
    gen.addAction(name="FORWARD[-2]", 
                  parameters = (("agent1", "agent"),("pt1" , "gridcell"),("pt2" , "gridcell"),("pt3" , "gridcell"),("t1" , "time"), ("t2", "time")), 
                  precondition_string = "(and (forward_next ?pt1 ?pt2)(forward_next ?pt2 ?pt3)(at ?pt1 agent1)(not (blocked ?pt2 ?t1))(not (blocked ?pt3 ?t1))(cur_time ?t1)(upd_time ?t1 ?t2))",
                  effect_string = "(and (at ?pt3 agent1)(not (at ?pt1 agent1))(not (cur_time ?t1))(cur_time ?t2))")
    gen.addAction(name="FORWARD[-3]", 
                  parameters = (("agent1", "agent"),("pt1" , "gridcell"),("pt2" , "gridcell"),("pt3" , "gridcell"),("pt4" , "gridcell"),("t1" , "time"), ("t2", "time")), 
                  precondition_string = "(and (forward_next ?pt1 ?pt2)(forward_next ?pt2 ?pt3)(forward_next ?pt3 ?pt4)(at ?pt1 agent1)(not (blocked ?pt2 ?t1))(not (blocked ?pt3 ?t1))(not (blocked ?pt4 ?t1))(cur_time ?t1)(upd_time ?t1 ?t2))",
                  effect_string = "(and (at ?pt4 agent1)(not (at ?pt1 agent1))(not (cur_time ?t1))(cur_time ?t2))")
    
    gen.generateDomainPDDL()
    pass

def generateProblemPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File.
    Objects defined here should be used to construct the init and goal strings
    '''
    gen.addProblemHeader("parking", "grid_world")
    gen.addObjects("agent", ["agent1"])
    time = []
    for t in range (gen.width+gen.num_lanes):
        time.append("t{}".format(t))
    gen.addObjects("time", time)
    gen.generateGridCells()
    gen.addObjects("gridcell", gen.grid_cell_list, isLastObject=True)
    gen.addInitState()
    gen.addGoalState()
    gen.generateProblemPDDL()
    pass

def runPDDLSolver(gen):
    '''
    Runs the fast downward solver to get the optimal plan
    '''
    os.system(FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH + 'fast-downward.py ' + PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name + ' ' + PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name + ' --search  \"lazy_greedy([ff()], preferred=[ff()])\"' + ' > temp ')

def delete_files(gen) :
    '''
    Deletes PDDL and plan files created.
    '''
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name)
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name)
    os.remove('sas_plan')


def simulateSolution(env):
    '''
    Simulates the plan given by the solver on the environment
    '''
    env.render()
    plan_file = open('sas_plan', 'r')
    for line in plan_file.readlines() :
#         print(line)
        if line[0] == '(' :
            action = line.split()[0][1:]
            print(action)
            if action == 'up' :
                env.step(env.actions[0])
            if action == 'down' :
                env.step(env.actions[1])
            if action == 'forward[-1]' :
                env.step(env.actions[4])
            if action == 'forward[-2]' :
                env.step(env.actions[3])
            if action == 'forward[-3]' :
                env.step(env.actions[2])
            env.render()

def generatePlan(env):
    '''
    Extracts the plan given by the solver into a list of actions
    '''
    plan_file = open('sas_plan', 'r')
    action_sequence = []
    for line in plan_file.readlines() :
        if line[0] == '(' :
            action = line.split()[0][1:]
            if action == 'up' :
                action_sequence.append(env.actions[0])
            if action == 'down' :
                action_sequence.append(env.actions[1])
            if action == 'forward[-1]' :
                action_sequence.append(env.actions[4])
            if action == 'forward[-2]' :
                action_sequence.append(env.actions[3])
            if action == 'forward[-3]' :
                action_sequence.append(env.actions[2])
    return action_sequence

def test() :
    '''
    Generates the PDDL files, solves for the optimal solution and simulates the plan. The PDDL files are deleted at the end.
    '''
    env=gym.make('GridDriving-v0', lanes=LANES, width=WIDTH, random_seed=RANDOM_SEED, agent_speed_range=(-3,-1))
    gen = initializeSystem(env)
    generateDomainPDDLFile(gen)
    generateProblemPDDLFile(gen)
    runPDDLSolver(gen)
    simulateSolution(env)
#     # delete_files(gen)

if SUBMISSION :
    from runner.abstracts import Agent
    class PDDLAgent(Agent):
        def initialize(self, params):
            global FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH
            FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = params[0]
            self.env = params[1]
            gen = initializeSystem(self.env)
            generateDomainPDDLFile(gen)
            generateProblemPDDLFile(gen)
            runPDDLSolver(gen)
            self.action_plan = generatePlan(self.env)
            self.time_step = 0
            delete_files(gen)

        def step(self, state, *args, **kwargs):
            action = self.action_plan[self.time_step]
            self.time_step +=1
            return action

    def create_agent(test_case_env, *args, **kwargs):
        return PDDLAgent() 
else :
    test()