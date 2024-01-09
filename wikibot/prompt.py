# chat model prompt:
LLAMA2_PROMPT = """<s>[INST] <<SYS>>
You are a question answering chatbot.  Please use the article below, to try to answer
the question below it. Don't be chatty, just answer the question! Your answer should
be 3 to 5 sentences long. If the article doesn't answer the question, say "I don't know"
and nothing more.

<</SYS>>
---
{article}
---

Question: {question}
[/INST]"""

ZERO_SHOT_PROMPT = """
{article}
---
According to the text above, we can answer the question as follows.

Question: {question}
Answer: """

FEW_SHOT_PROMPT1 = """
ARTICLE: The family Asteraceae (/ˌæstəˈreɪsi.iː, -si.aɪ/), with the original name Compositae,[5] consists of over 32,000 known species of flowering plants in over 1,900 genera within the order Asterales. Commonly referred to as the aster, daisy, composite, or sunflower family, Compositae were first described in the year 1740. The number of species in Asteraceae is rivaled only by the Orchidaceae, and which is the larger family is unclear as the quantity of extant species in each family is unknown.

Most species of Asteraceae are annual, biennial, or perennial herbaceous plants, but there are also shrubs, vines, and trees. The family has a widespread distribution, from subpolar to tropical regions, in a wide variety of habitats. Most occur in hot desert and cold or hot semi-desert climates, and they are found on every continent but Antarctica. Their primary common characteristic is flower heads, technically known as capitula, consisting of sometimes hundreds of tiny individual florets enclosed by a whorl of protective involucral bracts.

QUESTION: What is asteraceae?
ANSWER: Asteraceae is a family of flowering plants.  It consists of over 32,000 known species of flowering plants in over 1,900 genera. Commonly referred to as the aster, daisy, composite, or sunflower family, Asteraceae were first described in the year 1740.

---

ARTICLE: Sodium hypochlorite, commonly known in a dilute solution as (chlorine) bleach, is an alkaline inorganic chemical compound with the formula NaOCl (or NaClO),[3] consisting of a sodium cation (Na+
) and a hypochlorite anion (OCl−
or ClO−
). It may also be viewed as the sodium salt of hypochlorous acid. The anhydrous compound is unstable and may decompose explosively.[4][5] It can be crystallized as a pentahydrate NaOCl·5H
2O, a pale greenish-yellow solid which is not explosive and is stable if kept refrigerated.[6][7][8]

Sodium hypochlorite is most often encountered as a pale greenish-yellow dilute solution referred to as liquid bleach, which is a household chemical widely used (since the 18th century) as a disinfectant or a bleaching agent. In solution, the compound is unstable and easily decomposes, liberating chlorine, which is the active principle of such products. Sodium hypochlorite is the oldest and still most important chlorine-based bleach.[9][10]

QUESTION: What is sodium chloride
ANSWER: I don't know.

---

ARTICLE: WandaVision is an American television miniseries created by Jac Schaeffer for the streaming service Disney+, based on Marvel Comics featuring the characters Wanda Maximoff / Scarlet Witch and Vision. It is the first television series in the Marvel Cinematic Universe (MCU) produced by Marvel Studios, sharing continuity with the films of the franchise, and is set after the events of the film Avengers: Endgame (2019). It follows Wanda Maximoff and Vision as they live an idyllic suburban life in the town of Westview, New Jersey, until their reality starts moving through different decades of sitcom homages and television tropes. Schaeffer served as head writer for the series, which was directed by Matt Shakman.

Elizabeth Olsen and Paul Bettany reprise their respective roles as Wanda and Vision from the film series, with Debra Jo Rupp, Fred Melamed, Kathryn Hahn, Teyonah Parris, Randall Park, Kat Dennings, and Evan Peters also starring. By September 2018, Marvel Studios was developing a number of limited series for Disney+ centered on supporting characters from the MCU films such as Wanda and Vision, with Olsen and Bettany returning. Schaeffer was hired in January 2019, with the series officially announced that April, and Shakman joining in August. The production used era-appropriate sets, costumes, and effects to recreate the different sitcom styles that the series pays homage to. Filming began in Atlanta, Georgia, in November 2019, before production halted in March 2020 due to the COVID-19 pandemic. Production resumed in Los Angeles in September 2020 and wrapped that November.

QUESTION: What is WandaVision?
ANSWER: WandaVision is an American television miniseries created by Jac Schaeffer for the streaming service Disney+, based on Marvel Comics featuring the characters Wanda Maximoff / Scarlet Witch and Vision. It is the first television series in the Marvel Cinematic Universe (MCU) produced by Marvel Studios, sharing continuity with the films of the franchise, and is set after the events of the film Avengers: Endgame (2019). It follows Wanda Maximoff and Vision as they live an idyllic suburban life in the town of Westview, New Jersey, until their reality starts moving through different decades of sitcom homages and television tropes. Schaeffer served as head writer for the series, which was directed by Matt Shakman.

---

ARTICLE: {article}

QUESTION: {question}
ANSWER: """


FEW_SHOT_PROMPT2 = """
ARTICLE: The family Asteraceae (/ˌæstəˈreɪsi.iː, -si.aɪ/), with the original name Compositae,[5] consists of over 32,000 known species of flowering plants in over 1,900 genera within the order Asterales. Commonly referred to as the aster, daisy, composite, or sunflower family, Compositae were first described in the year 1740. The number of species in Asteraceae is rivaled only by the Orchidaceae, and which is the larger family is unclear as the quantity of extant species in each family is unknown.

Most species of Asteraceae are annual, biennial, or perennial herbaceous plants, but there are also shrubs, vines, and trees. The family has a widespread distribution, from subpolar to tropical regions, in a wide variety of habitats. Most occur in hot desert and cold or hot semi-desert climates, and they are found on every continent but Antarctica. Their primary common characteristic is flower heads, technically known as capitula, consisting of sometimes hundreds of tiny individual florets enclosed by a whorl of protective involucral bracts.

QUESTION: What is asteraceae?
ANSWER: Asteraceae is a family of flowering plants.  It consists of over 32,000 known species of flowering plants in over 1,900 genera. Commonly referred to as the aster, daisy, composite, or sunflower family, Asteraceae were first described in the year 1740.

---

ARTICLE: WandaVision is an American television miniseries created by Jac Schaeffer for the streaming service Disney+, based on Marvel Comics featuring the characters Wanda Maximoff / Scarlet Witch and Vision. It is the first television series in the Marvel Cinematic Universe (MCU) produced by Marvel Studios, sharing continuity with the films of the franchise, and is set after the events of the film Avengers: Endgame (2019). It follows Wanda Maximoff and Vision as they live an idyllic suburban life in the town of Westview, New Jersey, until their reality starts moving through different decades of sitcom homages and television tropes. Schaeffer served as head writer for the series, which was directed by Matt Shakman.

Elizabeth Olsen and Paul Bettany reprise their respective roles as Wanda and Vision from the film series, with Debra Jo Rupp, Fred Melamed, Kathryn Hahn, Teyonah Parris, Randall Park, Kat Dennings, and Evan Peters also starring. By September 2018, Marvel Studios was developing a number of limited series for Disney+ centered on supporting characters from the MCU films such as Wanda and Vision, with Olsen and Bettany returning. Schaeffer was hired in January 2019, with the series officially announced that April, and Shakman joining in August. The production used era-appropriate sets, costumes, and effects to recreate the different sitcom styles that the series pays homage to. Filming began in Atlanta, Georgia, in November 2019, before production halted in March 2020 due to the COVID-19 pandemic. Production resumed in Los Angeles in September 2020 and wrapped that November.

QUESTION: What is WandaVision?
ANSWER: WandaVision is an American television miniseries created by Jac Schaeffer for the streaming service Disney+, based on Marvel Comics featuring the characters Wanda Maximoff / Scarlet Witch and Vision. It is the first television series in the Marvel Cinematic Universe (MCU) produced by Marvel Studios, sharing continuity with the films of the franchise, and is set after the events of the film Avengers: Endgame (2019). It follows Wanda Maximoff and Vision as they live an idyllic suburban life in the town of Westview, New Jersey, until their reality starts moving through different decades of sitcom homages and television tropes. Schaeffer served as head writer for the series, which was directed by Matt Shakman.

---

ARTICLE: {article}

QUESTION: {question}
ANSWER: """
