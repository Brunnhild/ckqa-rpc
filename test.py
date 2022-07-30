from packages.completion.api import CompletionModel

if __name__ == '__main__':
    model = CompletionModel('./packages/completion/release')
    print(model.predict())
