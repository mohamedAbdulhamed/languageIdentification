from tkinter import *
from tkinter import ttk
import re
# import main


background = '#ccd3d6'
font = 'Cooper Std Black'
window = Tk()


# window.attributes('-fullscreen',True)
window.iconbitmap('natural-language-processing.ico')
window.winfo_toplevel().title("Auto Natural Language ID")
window.geometry("700x350")
window.config(background=background)

label = Label(window, text='Please enter a sentence to identify the language: ', font=(font, 14, 'bold'))
label.config(background=background)
label.pack(pady=20)
allowed_languages = ['عربي', 'English', 'Russian', 'Chinese', 'Hindi', 'Korean', 'Swedish', 'Tamil', 'Thai',
                     'Turkish']
string = ""
for lang in allowed_languages:
    string += f'{lang} - '
string = string[:-2]
label2 = Label(window, text=f'Allowed Languages: {string}', font=(font, 8))
label2.config(background=background)
label2.pack()

e = Entry(window, font=(font, 12))
e.config(width=50, background='#d1dee3')
e.pack(pady=20)
e.focus_force()


def sanitize_input(sentence):
    contractions = {
        "'ve": "have",
        "ain't": "am not / are not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is",
        "i'd": "I had / I would",
        "i'd've": "I would have",
        "i'll": "I shall / I will",
        "i'll've": "I shall have / I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }
    for word in sentence.split():
        if word.lower() in contractions:
            sentence = sentence.replace(word, contractions[word.lower()])
    return re.sub(r'[^A-Za-z]+', ' ', str(sentence).lower())


def get_lang(sentence):
    sanitized_input = sanitize_input(sentence)
    # main.fun(sanitized_input)
    return sanitized_input


def try_again():
    global result_label
    global try_btn
    e.delete(0, 'end')
    result_label.destroy()
    try_btn.destroy()
    if btn["state"] == 'disabled':
        btn["state"] = 'normal'


def click():
    global result_label
    global try_btn
    if btn["state"] == 'normal':
        btn["state"] = 'disabled'
    if not e.get() == '':
        result_label = Label(window, text=f'The language is: {get_lang(e.get())}', font=(font, 12))
        result_label.config(background=background)
        result_label.pack(pady=50)
    else:
        result_label = Label(window, text='No sentence has been entered!', font=(font, 12))
        result_label.config(background=background)
        result_label.pack(pady=50)
    try_btn = Button(window, text='Try again', command=try_again, font=(font, 10))
    try_btn.pack()


btn = Button(window, text='Check', command=click, font=(font, 10))
btn.pack()

separator = ttk.Separator(window, orient='horizontal')
separator.place(width=1000, rely=0.55)

window.mainloop()
