import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Класс Dataset для тестовых данных
class CEFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)


# Функция загрузки модели и метаданных
def load_model(model_path="cefr_model.pth", device="cpu"):
    # Убедимся, что модель загружается на правильное устройство
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=6
    ).to(device)

    # Загружаем веса с указанием устройства
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Загружаем токенизатор и label_encoder
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Model loaded successfully.")
    return model, tokenizer, label_encoder


# Функция получения предсказаний категорий для каждого объекта
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds


# Основной пайплайн
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка модели и метаданных
    model, tokenizer, label_encoder = load_model("cefr_model.pth", device=device)

    # Пример тестовых данных
    test_texts = ["Okay Laura. Let's go.Just a second, mom. Dont forget to put on your seatbelts. Do I have to? Yes. You have to. But we are only going to the store. It's the law, Laura. And to be safe, too. Right, mom?",

"""The Bayeux Tapestry
The Bayeux Tapestry (also known in France as a Tapestry of Queen Matilda) is a unique medieval artifact that dates back to the 11th century. Nearly 70 metres of embroidered1 cloth expand on the events that led up to the Norman conquest of England, culminating with the fateful Battle of Hastings.

Technically not a tapestry (as tapestries are woven, not embroidered), this exquisite piece of cloth shows about 70 historical scenes and is narrated with Latin tituli2. It’s origins and the history of creation are still hotly debated in scholarly circles, but the two main theories give the credit either to the Queen Matilda of Flanders who was a wife of William the Conqueror, or to a bishop Odo of Bayeux, who was William’s half-brother and eventually became a regent of England in his absence.

The tapestry is made largely of plain weave3 linen and embroidered with wool yarn. The woolen crewelwork4 is made in various shades of brown, blue and green, mainly terracotta, russet, and olive green. Later restorations have also added some brighter colours, such as orange and light yellow. Attempts at restoration of both the beginning and the end of the tapestry were made at some points, adding some missing tituli and numerals, although an ongoing debate disputes the validity of these restorations.

The events unfolding on a tapestry took place in the years 1064 to 1066. Anglo-Saxon earl Harold Godwinson is depicted receiving the English crown from Edward the Confessor, a deathly ill English monarch. An invading Norman force is then shown, which soon engages Saxon forces in a bloody battle. Ultimately king Harold is slain, and English forces flee the battlefield. The last part of the tapestry was supposedly lost and a newer piece was added in its place roughly in 1810.

The tapestry allows for an unique insight4 into the mind of a medieval craftsman, and, as it was commissioned by victorious Normans, gives us a chance to see how the medieval history was customarily5 chronicled by the winning side.

Since 1945 the Tapestry rests in Bayeux Museum, although as recently as 2018 the plans were put in motion to move it to an exhibit6 of the British Museum in London before the end of 2022. If everything proceeds as planned, it will be the first time the Tapestry has left France in over 950 years."
""",
"""Thus Spoke Zarathustra
‘Thus Spoke Zarathustra: A Book for All and None’ is a famous and somewhat controversial novel finalized by German philosopher Friedrich Nietzsche in 1885. Nietzsche has considered this book his most important work. It greatly expands on the main ideas that he has presented in his previous works, and remains a hot topic for debates in scholarly circles up to this day.

The book was written in German, and made heavy use of various forms of wordplay. The translations were thus sometimes impeded1 by a lack of corresponding wordplays or terms in other languages. Even taken at face value, the book was made explicitly2 in a way that defies3 any attempts to read it straightforwardly. Nietzsche himself, rather tongue-in-cheek4, has written thus in a preface to his next book, Ecce Homo: ‘With Thus Spoke Zarathustra I have given mankind the greatest present that has ever been made to it so far. This book, with a voice bridging centuries, is not only the highest book there is, the book that is truly characterized by the air of the heights — the whole fact of man lies beneath it at a tremendous distance — it is also the deepest, born out of the innermost wealth of truth, an inexhaustible well to which no pail descends without coming up again filled with gold and goodness’, perhaps hinting at the fact that none of his contemporaries5 had even begun to move in the right direction regarding that book.

The plot of the book is fairly simple. Zarathustra, a wandering philosopher, travels around the world and comments on various people and places he sees. Zarathustra is an evaluator (or rather, transvaluator) of all ideas, and strives to question a broad variety of topics regarding human culture and daily lives.

Three major themes can be followed through the book: the eternal recurrence of everything that is; the possible appearance of ‘super-humanity’; the concept of ‘will to power’ as the cornerstone of human psyche and behaviour.

The idea of ‘eternal return’ (or recurrence) is the idea that each event and occurrence that happens, repeats itself eternally in cycles. Rather than postulating this, Nietzsche actually ponders if it’s true. Although it’s a very popular idea that seemingly stems logically from the laws of infinite Universe as we know it, it still hasn’t been proven nor disproven, so Nietzsche marks it as ‘the most burdensome’ of his thoughts.

The concept of a ‘super-human’ (or, rather, of a ‘beyond-human’, Übermensch) is one of the goals that Nietzsche suggests to humanity through teachings of Zarathustra. The Übermensch is an objectively better type of a human that is destined to transcend6 the regular humans. This idea was interpreted in wildly different ways, sometimes outright xenophobic. But at its core it suggests only transcendence of some stale norms of morality and building a better future on Earth instead of turning to all things spiritual. An antithesis of an Übermensch is called a ‘last man’, a nihilistic, egalitarian and decadent human being, ‘too apathetic to dream’. Nietzsche also suggests that this is another of the possible outcomes of humanity development.

The third idea, which is a ‘will to power’ is never precisely defined in any of Nietzche’s work. This also has brought many speculations and controversy into his works, as well as into the works of his researchers. He did mention though that it’s a driving characteristic of all life, and it’s related to overcoming perils7 and obstacles, including the obstacles within oneself. He also made a notion that human cruelty (in whatever form) may be related to this driving force.

Initially Nietzsche has planned this book to have six parts. During his life he’s managed to write only four, and the fourth was largely written as a rough draft. Debates around the book are still going strong today, and while Nietzsche himself has argued that the book is finished, and opposed vehemently to any attempts to add or remove something from it, the key to the ultimate understanding of his ideas is yet to be found."
""",
"""Baby K
The development of a human embryo can go awry1 in many different ways. One of the most common types of birth defects that afflict yet unborn children are referred to as neural tube defects (NTDs). A premise for the development of NTDs lies in an incomplete closure of a neural tube, a precursor2 to the human central nervous system that forms from an embryo’s nervous tissue3 over the course of a normal development. As a result, an opening remains in the developing spine or cranium4 of the fetus, which, depending on its severity, can fully disrupt the growth of the nervous system. Neural tube defects affect either the development of the brain, or spine, or both. Most of the conditions that stem from NTDs are usually untreatable, leave the person largely disabled, and have an extremely high mortality rate.

Anencephaly is a NTD that in broadest terms means the complete absence of the cerebrum5, the largest part of the brain responsible for senses and cognition. The causes of the condition are still unclear, but it is speculated that it can be triggered by a folic acid deficiency and certain types of diabetes in pregnant women. Abortion is strongly encouraged when anencephaly is detected via ultrasound. Anencephalic children are usually either stillborn6, or die from cardiorespiratory arrest mere hours or a few days after the birth.

Nevertheless, there were some cases of anencephaly that truly stood out from the rest. One of such cases was of Stephanie Keene (name was probably changed due to privacy concerns) dubbed as Baby K.

Stephanie was diagnosed with anencephaly long before her birth. Her mother has chosen to keep the child due to her belief as a Christian that all life should be protected.

The doctors and the nurses both strongly advocated for a DNR order7 for the baby, but the mother refused yet again. Over the course of six months after the birth Stephanie has travelled from hospital to a hospital and was kept under a ventilator all this time. Eventually a hospital has filed a lawsuit against Stephanie’s mother, aiming to appoint a legal guardian in her place, and trying to receive a legal confirmation that the hospital couldn’t be held responsible for Stephanie’s health and would opt out8 of any services to her save for palliative caregiving.

And, in a very controversial ruling, the hospital has lost that case. The court has ruled that Stephanie is to be put under a mechanical ventilator and be given other care if any sort of other medical emergency would have arisen. The court has also made a notion that they ruled according to existing laws, without any regard to the rather unusual condition that Stephanie had.

Thus Baby K has lived 2 years and 174 days. Her heart had stopped on April 5, 1995. Keeping her heart beating had cost over 500,000$, a sum, as some would argue, that could’ve been spent on research aimed to prevent NTDs or, possibly, treatment of other newborn children."
""",
"""A murder mystery as a literary genre
A crime fiction (also called a murder mystery) is a story that focuses on a criminal act and on a following investigation1. Usually done from a point of view of either a detective or their assistant, crime fiction spans over many types of media. Usually it takes the form of either a novel or a movie.

The first historical example of crime fiction is probably a novel The Three Apples. It was a part of One Thousand and One Nights, which is a collection of old Arabic folk tales. The novel lacked any typical features of a modern murder mystery, but still tried to set up a crime scene as a plot2 device. Other tales from this collection also describe some bits and pieces3 of actual crime investigation.

The genre became very popular in the late 19th century, with works by Edgar Allan Poe and Arthur Conan Doyle paving4 the way for more advanced stories of John Dickson Carr and Agatha Christie. Sherlock Holmes and Hercule Poirot, while being purely fictional characters, became real enough to their own fans. Over the course of many years readers were following the adventures of their beloved detectives. Holmes has appeared in 60 works of fiction in total, while Poirot in his career has made over 80 appearances.

A classic murder mystery can be viewed as a sort of a game between an author and the reader. An author sets up a murder scene, and the reader must deduce5 the culprit6 before the main detective character reveals him. A typical murder mystery leaves three questions to the reader: who has done it? How was it done? Why was it done? Answering all three questions before the main character would mean ‘beating’ the novel.

As the genre developed further, authors have developed some guidelines on writing a good murder mystery. There were many variations of such rules, but in a nutshell7 it all boiled down to a novel being fair to its reader. For example, a good novel had to introduce the culprit early in the story as someone who a reader would know about. All clues should be available to the reader the same way they are available for the protagonists. There were also some very strict rules on the usage of poison and other similar substances, as the reader should have been able to unravel8 the story without any sort of special knowledge.

One of the most iconic form of a murder fiction is the locked-room mystery, which describes seemingly an impossible crime (for example, a corpse would be hidden inside an empty room that is locked from the inside) and challenges the reader to find a plausible way to explain it and eventually find the perpetrator9.

Another type of murder novels revolves around a closed circle of suspects. These stories usually have many colorful characters, each of them with their own agenda10, and the main challenge for a reader lies in pointing out the single guilty party while sparing the rest of possible culprits.

The murder mystery is still a very popular genre nowadays, and the classics of it are routinely adapted into films, videogames and some other forms of fan fiction."
""",
""""Global consequences of the climate change
The 20th century was very notable with its unparalleled1 technological advancement of humanity. With each passing day the lasting impact that we leave on our planet becomes more and more apparent2. The most obvious and harmful outcomes of heavy industrialization are global warming and climate change.

The first signs of global warming became obvious in the middle of the last century. Since the 1970s, the surface temperature of Earth has risen by 1 °C. Multiple data records show now that the warming happens at the rate of roughly 0.2 °C per one decade.

This is a very alarming development. The bulk3 of global warming is attributed to human activity. Assuming we don’t do something about it, the consequences would be lasting, probably irreversible4, and very harsh.

The first and most obvious effect is the heating of Earth’s atmosphere. This means that there will be less cold days and more hot days overall5. This in turn means that both plants and animals will need to adjust to it. Some of them might not survive such a change.

The secondary effect is the melting of continental ice, which makes sea levels rise far above their normal point. Extreme cases could lead to floods and destruction of continental coastlines.

Warmer weather also results in more water evaporating and the air becoming more humid6. This can lead to even more rains, floods and some extreme weather patterns such as wildfires and tropical cyclones.

One of the most insidious7 and less obvious effects is the change of the oceans oxygen levels. Warmer water can hold less oxygen than the colder one, and so if the temperatures continue to rise, many underwater species risk total extinction.

While humanity definitely contributes much to climate change with irresponsible8 burning of fossil fuels, we still can battle it. Switching to renewable and clear energy sources, electrical cars, and improving the efficiency of our factories can curb the adverse effects we’ve inflicted on our planet over the last 100 years.

And if worse comes to worst9, humanity can be very good at adapting to hostile environments10. Adaptation strategies include reinforcing the coastlines or relocating deeper into the mainland; development of weather-resistant crops; development of contingency11 scenarios for local disaster management."
""",
                  "I live in city.I have dog.Dog is big.I like dog",
                  "Yesterday I went to park with my friend.We played football and ate ice cream.",
    """If I had more time, I would travel more often.Last year I visited three countries.""",
    """Despite the economic challenges, many companies continue
    to
    invest in renewable
    energy
    sources.""",
    """The
    proliferation
    of
    digital
    technologies
    has
    precipitated
    a
    paradigm
    shift in educational
    methodologies""",
    """
    Contemporary
    neuroscientific
    research
    has
    elucidated
    the
    hitherto
    obscure
    mechanisms
    underlying
    cognitive
    processes."""

]
    test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Важно использовать правильные метки

    # Инициализация тестового датасета и DataLoader
    test_dataset = CEFRDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Получение предсказаний категорий для каждого объекта
    predictions = get_predictions(model, test_loader, device)

    # Выводим результаты
    for text, prediction in zip(test_texts, predictions):
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        print(f"Text: '{text}' -> Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
