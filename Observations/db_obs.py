from Observations.obs import obs
import numpy as np

class DBObservation(obs):
    pass

class DBObservationProto(DBObservation):
 
  def __init__(self,db_conn,states_before, actions, states_after,model_estimaitor):
    self.db_conn = db_conn
    self.states_before = states_before
    self.actions = actions
    self.states_after = states_after
    self.model_estimaitor = model_estimaitor
    self.state_dim = len(states_before[0])
    self.total_states = len(states_before)
    self.CLIP_MIN = 0
    self.CLIP_MAX = 100

  def get_state_space(self):
        state_id = self.db_conn.state_id
        action_ =  self.db_conn.current_action
       
        # If the taken action matches recorded action → return true next state
        if self.actions[state_id] == action_:
            print(self.actions[state_id])
            return np.array(self.states_after[state_id], dtype=np.float32)
        else:
            # Predict next state with external model (defined outside)
            state_before_i = np.array(self.states_before[state_id])
            print("state_before_i",state_before_i)
            action_ = np.array(action_)
            print("action_",action_)
            model_prediction = self.model_estimaitor.predict([state_before_i.reshape(1, -1), action_.reshape(1, -1)],verbose=0)
            print("model_prediction",model_prediction)
            return np.clip(model_prediction.flatten(), self.CLIP_MIN, self.CLIP_MAX).astype(np.float32).flatten()

  
  def len_sensors(self):
    return 7