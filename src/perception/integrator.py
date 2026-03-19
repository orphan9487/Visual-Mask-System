import collections

class ContextualBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add_message(self, role, message):
        self.buffer.append({"role": role, "content": message})

    def get_history(self):
        return list(self.buffer)

class M_ICL_Integrator:
    def __init__(self):
        self.template = "History Context: {history}\nCurrent Input: {tokens}\nSynthesized Output:"

    def synthesize(self, salient_tokens, history):
        history_str = " | ".join([f"{m['role']}: {m['content']}" for m in history])
        synthesized_context = self.template.format(
            history=history_str, 
            tokens=salient_tokens
        )
        return synthesized_context

if __name__ == "__main__":
    buffer = ContextualBuffer(buffer_size=3)
    integrator = M_ICL_Integrator()
    
    buffer.add_message("User", "I'm so tired of this.")
    buffer.add_message("System", "Take a rest then.")
    
    current_tokens = "forgot the keys again. unbelievable"
    history = buffer.get_history()
    
    final_context = integrator.synthesize(current_tokens, history)
    print(final_context)

    