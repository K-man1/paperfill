# PaperFill
Paperfill is an AI PDF filler which finds the blanks that are in your PDF (open space, fill-in-the-blanks, tables, etc) and uses its knowledge to fill it in. It also has handwriting support, so you can just upload a template of your handwriting and it copies it. 

It's unique from Claude and other LLM's PDF editing tools and other AI PDF editors because it doesnt use a coordinate based system making sure all the text is in the correct spot while Claude often messes up and the words are in the middle of the blank. 


Paperfill: <img width="645" height="78" alt="paperfill" src="https://github.com/user-attachments/assets/9e71ad5c-9ab8-4b30-a299-618af4ee6c61" />

Claude: <img width="628" height="78" alt="Screenshot 2026-07-08 at 6 43 28 PM" src="https://github.com/user-attachments/assets/85ee9e69-56e8-4269-9d79-e1f1cd08fafc" />

I personally use it for my HW (dont tell me teachers 😭) and for quickly completing the low effort clearly-created-with-ChatGPT worksheets that my spanish teacher assigns.

Use it here: [karman.hackclub.app](karman.hackclub.app)

See the demo: [https://www.youtube.com/watch?v=bDNsQvA0_DU&feature=youtu.be](demo) plz dont judge the cough

## Features

- Editing
  - I knew it wasnt gunna be perfect ALL the time, so I added the ability to add, edit, and remove text boxes
  - You can also edit text boxes with AI. Just select a text box and choose lengthen, shorten, or give it another prompt
- AI screenshots
  - So I also knew if ur getting AI to fill the PDF for you, sometimes you may not know the info. that means you can take a "screenshot" of the question and AI will answer it and you can move the answer to the correct location.
- Handwriting
  - Simply upload a filled template of your handwriting, and then you can have PDFs filled in with what looks like your handwriting.(https://github.com/yashlamba/handwrite)

## Prerequisites

Python 3.10+

potrace system binary (only needed for the handwriting-font feature)

macOS: brew install potrace

Debian/Ubuntu: apt-get install potrace

## Setup
``` 
git clone https://github.com/K-man1/paperfill
cd PaperFill

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
