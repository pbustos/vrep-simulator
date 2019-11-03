from inputs import devices, get_gamepad
import threading

class JoyReader(threading.Thread):
    MAX_ZADV = 20 #mm/sg
    MAX_XADV = 20 #mm/sg
    MAX_YADV = 20 #mm/sg
    end = False
    unloading = False
    next_ep = False
    incs = [0.0, 0.0, 0.0]
    def __init__(self):
	    super(JoyReader, self).__init__(daemon= True)

    def run(self):
        while(True):
            event = get_gamepad()[0]
            #print(event.code,  event.state, event.ev_type)
            if event.code == "ABS_Y":
                self.incs[2] = -(-event.state*self.MAX_ZADV*2/256 + self.MAX_ZADV) / 1000.      # up-down
            elif event.code == "ABS_X":
                self.incs[0] = -(-event.state*self.MAX_XADV*2/256 + self.MAX_XADV) / 1000.      # left-right
            elif event.code == "ABS_RZ":
                self.incs[1] = (-event.state*self.MAX_YADV*2/256 + self.MAX_YADV) / 1000.      # foward-backward
            elif event.code == "BTN_THUMB" and event.state == 1:
                self.next_ep = True
            elif event.code == "BTN_BASE5":
                self.end = True
            elif event.code == "BTN_TRIGGER" and event.state == 1:
                self.unloading = True
            elif event.code == "BTN_TRIGGER" and event.state == 0:
                self.unloading = False   