"""Scaled-up relation/fact set for the decoherence robustness check --
6 relations x 20 facts = 120 facts, vs. the original 3 x 15 = 45. Same
single-sentence template style as verifier_multi_relation_robustness.RELATIONS
(kept for the "small" comparison point); relations_large_docs.py builds the
multi-paragraph "realistic RAG document" variant on top of this same fact set.
"""
from __future__ import annotations

RELATIONS_LARGE = {
    "capital_of": {
        "doc": "{h} is a country whose capital is {t}.",
        "q": "What is the capital of {h}?",
        "facts": [
            ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Japan", "Tokyo"),
            ("China", "Beijing"), ("Russia", "Moscow"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
            ("Brazil", "Brasilia"), ("India", "New Delhi"), ("Greece", "Athens"), ("Poland", "Warsaw"),
            ("Sweden", "Stockholm"), ("Turkey", "Ankara"), ("Argentina", "Buenos Aires"),
            ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Austria", "Vienna"),
            ("Thailand", "Bangkok"),
        ],
    },
    "founded_by": {
        "doc": "{h} is a company founded by {t}.",
        "q": "Who founded {h}?",
        "facts": [
            ("Microsoft", "Bill Gates"), ("Facebook", "Mark Zuckerberg"), ("Amazon", "Jeff Bezos"),
            ("Tesla", "Elon Musk"), ("Apple", "Steve Jobs"), ("Dell", "Michael Dell"),
            ("Oracle", "Larry Ellison"), ("Nike", "Phil Knight"), ("IKEA", "Ingvar Kamprad"),
            ("Ford", "Henry Ford"), ("Disney", "Walt Disney"), ("Netflix", "Reed Hastings"),
            ("Twitter", "Jack Dorsey"), ("LinkedIn", "Reid Hoffman"), ("Airbnb", "Brian Chesky"),
            ("Spotify", "Daniel Ek"), ("Instagram", "Kevin Systrom"), ("Snapchat", "Evan Spiegel"),
            ("Reddit", "Steve Huffman"), ("Pinterest", "Ben Silbermann"),
        ],
    },
    "written_by": {
        "doc": "{h} is a book written by {t}.",
        "q": "Who wrote {h}?",
        "facts": [
            ("1984", "George Orwell"), ("Hamlet", "William Shakespeare"), ("Dracula", "Bram Stoker"),
            ("Frankenstein", "Mary Shelley"), ("Moby Dick", "Herman Melville"),
            ("Pride and Prejudice", "Jane Austen"), ("War and Peace", "Leo Tolstoy"),
            ("The Odyssey", "Homer"), ("Don Quixote", "Miguel de Cervantes"),
            ("Crime and Punishment", "Fyodor Dostoevsky"), ("The Great Gatsby", "F. Scott Fitzgerald"),
            ("Great Expectations", "Charles Dickens"), ("Wuthering Heights", "Emily Bronte"),
            ("The Hobbit", "J.R.R. Tolkien"), ("Brave New World", "Aldous Huxley"),
            ("Lolita", "Vladimir Nabokov"), ("Ulysses", "James Joyce"), ("Dune", "Frank Herbert"),
            ("The Trial", "Franz Kafka"), ("Beloved", "Toni Morrison"),
        ],
    },
    "painted_by": {
        "doc": "{h} is a painting created by {t}.",
        "q": "Who painted {h}?",
        "facts": [
            ("The Mona Lisa", "Leonardo da Vinci"), ("The Starry Night", "Vincent van Gogh"),
            ("Guernica", "Pablo Picasso"), ("The Scream", "Edvard Munch"),
            ("The Persistence of Memory", "Salvador Dali"), ("The Birth of Venus", "Sandro Botticelli"),
            ("American Gothic", "Grant Wood"), ("Girl with a Pearl Earring", "Johannes Vermeer"),
            ("The Night Watch", "Rembrandt"), ("Water Lilies", "Claude Monet"),
            ("The Last Supper", "Leonardo da Vinci"), ("Las Meninas", "Diego Velazquez"),
            ("The Kiss", "Gustav Klimt"), ("Nighthawks", "Edward Hopper"),
            ("Composition VIII", "Wassily Kandinsky"), ("The Garden of Earthly Delights", "Hieronymus Bosch"),
            ("Whistler's Mother", "James McNeill Whistler"), ("The Hay Wain", "John Constable"),
            ("Impression Sunrise", "Claude Monet"), ("The Third of May 1808", "Francisco Goya"),
        ],
    },
    "directed_by": {
        "doc": "{h} is a film directed by {t}.",
        "q": "Who directed {h}?",
        "facts": [
            ("Jaws", "Steven Spielberg"), ("Titanic", "James Cameron"), ("Psycho", "Alfred Hitchcock"),
            ("Pulp Fiction", "Quentin Tarantino"), ("Inception", "Christopher Nolan"),
            ("The Godfather", "Francis Ford Coppola"), ("Jurassic Park", "Steven Spielberg"),
            ("Avatar", "James Cameron"), ("Casablanca", "Michael Curtiz"),
            ("The Shining", "Stanley Kubrick"), ("Goodfellas", "Martin Scorsese"),
            ("Schindler's List", "Steven Spielberg"), ("Vertigo", "Alfred Hitchcock"),
            ("Fight Club", "David Fincher"), ("The Matrix", "Lana Wachowski"),
            ("Apocalypse Now", "Francis Ford Coppola"), ("Alien", "Ridley Scott"),
            ("Blade Runner", "Ridley Scott"), ("Taxi Driver", "Martin Scorsese"),
            ("E.T. the Extra-Terrestrial", "Steven Spielberg"),
        ],
    },
    "composed_by": {
        "doc": "{h} is a musical work composed by {t}.",
        "q": "Who composed {h}?",
        "facts": [
            ("The Fifth Symphony", "Ludwig van Beethoven"), ("The Magic Flute", "Wolfgang Amadeus Mozart"),
            ("The Nutcracker", "Pyotr Ilyich Tchaikovsky"), ("The Four Seasons", "Antonio Vivaldi"),
            ("Messiah", "George Frideric Handel"), ("The Rite of Spring", "Igor Stravinsky"),
            ("Carmen", "Georges Bizet"), ("Swan Lake", "Pyotr Ilyich Tchaikovsky"),
            ("Symphony No. 9", "Ludwig van Beethoven"), ("Requiem", "Wolfgang Amadeus Mozart"),
            ("The Barber of Seville", "Gioachino Rossini"), ("Peer Gynt", "Edvard Grieg"),
            ("Boléro", "Maurice Ravel"), ("Clair de Lune", "Claude Debussy"),
            ("The Planets", "Gustav Holst"), ("Madame Butterfly", "Giacomo Puccini"),
            ("Aida", "Giuseppe Verdi"), ("Fantasia on Greensleeves", "Ralph Vaughan Williams"),
            ("Symphony No. 40", "Wolfgang Amadeus Mozart"), ("Pictures at an Exhibition", "Modest Mussorgsky"),
        ],
    },
}
