"""The fixed retention probe: 50 general-knowledge sentences.

The forgetting guard measures mean NLL over these after every nightly
update. The set is deliberately broad (science, math, programming,
history, geography, language, everyday reasoning) and **frozen**: it must
never be edited once real runs begin, or retention numbers stop being
comparable across nights and sweeps. Add a v2 list instead if needed.
"""

RETENTION_PROBE_V1: list[str] = [
    # -- science ------------------------------------------------------------
    "The Earth orbits the Sun once every 365.25 days, which is why leap years "
    "exist: an extra day every four years keeps the calendar aligned.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen, "
    "using energy captured from sunlight by chlorophyll.",
    "The freezing point of water at standard pressure is zero degrees Celsius, "
    "or thirty-two degrees Fahrenheit.",
    "Sound travels through air at roughly 343 meters per second, which is why "
    "thunder arrives seconds after the lightning that caused it.",
    "DNA stores genetic information as sequences of four bases: adenine, "
    "thymine, guanine, and cytosine.",
    "An object in motion stays in motion unless acted on by an outside force; "
    "this is Newton's first law of motion.",
    "Atoms consist of a nucleus of protons and neutrons surrounded by "
    "electrons, and the number of protons determines the element.",
    "Antibiotics kill bacteria or stop their growth, but they have no effect "
    "on viruses, which is why they do not cure the common cold.",
    "The human heart has four chambers: two atria that receive blood and two "
    "ventricles that pump it out to the lungs and body.",
    "Light from the Sun takes a little over eight minutes to reach the Earth.",
    # -- mathematics ---------------------------------------------------------
    "A binary search over a sorted array of one million elements needs at most "
    "twenty comparisons, because two to the twentieth exceeds one million.",
    "The sum of the interior angles of any triangle in a plane is always one "
    "hundred eighty degrees.",
    "A prime number is a whole number greater than one whose only divisors are "
    "one and itself; the smallest primes are two, three, five, and seven.",
    "Multiplying any number by zero gives zero, and adding zero to any number "
    "leaves it unchanged.",
    "The area of a circle equals pi times the square of its radius, where pi "
    "is approximately 3.14159.",
    "Compound interest grows a balance exponentially: money earning ten "
    "percent per year roughly doubles in about seven years.",
    "The square root of 144 is 12, because 12 multiplied by itself equals 144.",
    "In a right triangle, the square of the hypotenuse equals the sum of the "
    "squares of the other two sides; this is the Pythagorean theorem.",
    "The median of a sorted list is its middle value, and unlike the mean it "
    "is robust to a few extreme outliers.",
    "Two to the tenth power is 1024, which is why a kibibyte contains 1024 "
    "bytes rather than an even thousand.",
    # -- programming ----------------------------------------------------------
    "In Python, dictionaries preserve insertion order since version 3.7, and "
    "lookups by key take constant time on average.",
    "A stack is last-in first-out while a queue is first-in first-out; "
    "function calls use a stack, print jobs use a queue.",
    "Git records the history of a project as a graph of commits, and a branch "
    "is simply a movable pointer to one of those commits.",
    "In most programming languages, array indices start at zero, so the first "
    "element of an array a is a[0].",
    "HTTP status code 404 means the requested resource was not found, while "
    "500 signals an internal server error.",
    "A SQL SELECT statement reads rows from a table, and adding a WHERE clause "
    "filters which rows are returned.",
    "Unit tests catch regressions early by automatically verifying each "
    "component's expected behavior after every change.",
    "JSON represents structured data with objects in curly braces, arrays in "
    "square brackets, and strings in double quotes.",
    "Regular expressions describe text patterns; the expression a+ matches one "
    "or more consecutive letter a characters.",
    "Hash tables trade memory for speed, offering average constant-time "
    "insertion and lookup by storing values in buckets chosen by a hash function.",
    # -- history & geography ---------------------------------------------------
    "The Great Wall of China was built over many centuries to defend against "
    "invasions from the north.",
    "The first humans landed on the Moon in 1969 aboard Apollo 11, and Neil "
    "Armstrong was the first to step onto its surface.",
    "The capital of France is Paris, the capital of Japan is Tokyo, and the "
    "capital of Australia is Canberra.",
    "The Amazon is the largest rainforest on Earth and its river carries more "
    "water than any other river in the world.",
    "World War II ended in 1945, after which the United Nations was founded to "
    "promote international cooperation.",
    "The printing press, introduced in Europe by Johannes Gutenberg in the "
    "fifteenth century, made books dramatically cheaper to produce.",
    "Mount Everest, on the border between Nepal and China, is the tallest "
    "mountain above sea level on Earth.",
    "The Sahara is the largest hot desert in the world, stretching across "
    "much of northern Africa.",
    "Ancient Rome's republic gave way to an empire under Augustus, the first "
    "Roman emperor.",
    "The Pacific is the largest and deepest ocean, covering about a third of "
    "the Earth's surface.",
    # -- language & everyday reasoning -----------------------------------------
    "A synonym is a word with nearly the same meaning as another, while an "
    "antonym means the opposite.",
    "In English, the past tense of go is went, an irregular form that does "
    "not follow the usual -ed pattern.",
    "If a recipe serves four and you need to serve eight, you double every "
    "ingredient.",
    "2.5 hours is 150 minutes, because each hour contains 60 minutes.",
    "When ice melts into water its mass stays the same, even though its "
    "volume decreases slightly.",
    "A dozen means twelve, so three dozen eggs is thirty-six eggs.",
    "If a train departs at 9:40 and the ride lasts 45 minutes, it arrives at "
    "10:25.",
    "Reading a thermometer in the shade gives a more accurate air temperature "
    "than reading it in direct sunlight.",
    "To alphabetize the words banana, apple, and cherry, the correct order is "
    "apple, banana, cherry.",
    "Borrowing money costs more than paying cash when interest accrues, "
    "because the borrower repays the principal plus interest.",
]
