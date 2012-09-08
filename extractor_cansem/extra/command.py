import subprocess, threading
import time
import os, signal

class Command(object):
    '''
    Call subprocess. Wait. If it doesn't return within the agreed timeframe: kill it
    '''
    def __init__(self, cmd):
        self.cmd = cmd
        #self.process = None

    def target(self):
        print 'Thread started'
        self.process = subprocess.Popen(self.cmd, shell=False, env=os.environ)
        self.process.communicate()
        print 'Thread finished'
    
    def run(self, timeout):

        thread = threading.Thread(target=self.target)
        thread.start()
        time.sleep(1)
        thread.join(timeout)
        if thread.is_alive():
            print 'Terminating process!'
            print self.process.pid
            try:
                print self.process.pid
                self.process.terminate()
            except:
                print "Process no longer exists"
            #time.sleep(3)
            thread.join()
            return False
        return True

#command = Command("echo 'x'; sleep 2; echo 'y'")
#command.run(timeout=3)
#command.run(timeout=1)


