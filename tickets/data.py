"""Synthetic support tickets in English and German, labelled with category and priority.

Real helpdesk tickets contain customer data, so this module generates realistic ones instead:
six categories, two languages, typos, off-topic sentences that mention other categories, and
8% label noise on priority (people do not label tickets consistently either).
"""
from __future__ import annotations

import random

CATEGORIES = ["account_access", "billing", "bug", "network", "hardware", "feature_request"]
PRIORITIES = ["low", "medium", "high"]

PROBLEMS = {
    "account_access": {
        "en": ["I can't log in to my account", "my password reset link does not work", "I am locked out after too many attempts",
               "two-factor authentication code is never accepted", "I need access to the {app} admin area",
               "my account was deactivated without notice", "SSO login redirects me back to the start page"],
        "de": ["ich kann mich nicht in mein Konto einloggen", "der Link zum Zurücksetzen des Passworts funktioniert nicht",
               "mein Konto ist nach zu vielen Versuchen gesperrt", "der Zwei-Faktor-Code wird nicht akzeptiert",
               "ich brauche Zugriff auf den Admin-Bereich von {app}", "mein Benutzerkonto wurde deaktiviert"],
    },
    "billing": {
        "en": ["I was charged twice for my subscription", "the invoice for {month} shows the wrong amount",
               "please refund the last payment", "my credit card payment failed but money was taken",
               "I need a copy of invoice INV-{num}", "the VAT number is missing on my invoice"],
        "de": ["mir wurde das Abo doppelt berechnet", "die Rechnung für {month} hat den falschen Betrag",
               "bitte erstatten Sie die letzte Zahlung", "die Kartenzahlung ist fehlgeschlagen, aber das Geld wurde abgebucht",
               "ich brauche eine Kopie der Rechnung INV-{num}", "auf meiner Rechnung fehlt die USt-IdNr"],
    },
    "bug": {
        "en": ["the export button in {app} throws an error", "the app crashes when I upload a file", "the dashboard shows wrong numbers since the update",
               "saving the form gives error 500", "the search returns no results even for existing entries",
               "the page freezes when I open the reports tab"],
        "de": ["der Export-Button in {app} zeigt einen Fehler", "die App stürzt beim Hochladen einer Datei ab",
               "das Dashboard zeigt seit dem Update falsche Zahlen", "beim Speichern des Formulars kommt Fehler 500",
               "die Suche findet keine Ergebnisse", "die Seite friert ein, wenn ich den Berichte-Tab öffne"],
    },
    "network": {
        "en": ["the VPN disconnects every few minutes", "Wi-Fi in the {office} office is very slow", "I cannot reach the internal server",
               "the connection to the database times out", "our site-to-site link is down", "DNS does not resolve our intranet"],
        "de": ["das VPN trennt sich alle paar Minuten", "das WLAN im Büro {office} ist sehr langsam", "ich erreiche den internen Server nicht",
               "die Verbindung zur Datenbank läuft in einen Timeout", "die Standortverbindung ist ausgefallen", "das DNS löst das Intranet nicht auf"],
    },
    "hardware": {
        "en": ["my laptop does not turn on", "the printer on floor {num} is jammed", "my monitor flickers constantly",
               "the docking station does not detect my laptop", "the keyboard stopped working", "the battery drains within an hour"],
        "de": ["mein Laptop geht nicht mehr an", "der Drucker im {num}. Stock hat einen Papierstau", "mein Monitor flackert ständig",
               "die Dockingstation erkennt meinen Laptop nicht", "die Tastatur funktioniert nicht mehr", "der Akku ist nach einer Stunde leer"],
    },
    "feature_request": {
        "en": ["it would be great to export reports as CSV", "could you add a dark mode to {app}", "please add an option to schedule emails",
               "we would like an API endpoint for invoices", "can you support German in the user interface", "please allow bulk editing of entries"],
        "de": ["es wäre toll, Berichte als CSV exportieren zu können", "könnt ihr einen Dark Mode für {app} einbauen",
               "bitte eine Option zum Planen von E-Mails hinzufügen", "wir hätten gern einen API-Endpunkt für Rechnungen",
               "bitte die Mehrfachbearbeitung von Einträgen erlauben"],
    },
}

# Things people complain about, per category. Combined with the generic patterns below, so the
# model has to learn the domain words rather than memorise whole sentences.
OBJECTS = {
    "account_access": {"en": ["my login", "the password reset", "my user account", "the 2FA code", "single sign-on", "my access rights", "the account lockout"],
                       "de": ["mein Login", "das Zurücksetzen des Passworts", "mein Benutzerkonto", "der 2FA-Code", "das Single Sign-on", "meine Zugriffsrechte", "die Kontosperre"]},
    "billing": {"en": ["my invoice", "the refund", "the subscription charge", "the credit card payment", "the VAT on my bill", "the billing address", "the direct debit"],
                "de": ["meine Rechnung", "die Rückerstattung", "die Abo-Abbuchung", "die Kreditkartenzahlung", "die Mehrwertsteuer auf der Rechnung", "die Rechnungsadresse", "die Lastschrift"]},
    "bug": {"en": ["the export feature", "the file upload", "the reports page", "the search function", "saving a form", "the dashboard chart", "the latest app update"],
            "de": ["die Exportfunktion", "der Datei-Upload", "die Berichtsseite", "die Suchfunktion", "das Speichern eines Formulars", "das Dashboard-Diagramm", "das letzte App-Update"]},
    "network": {"en": ["the VPN", "the office Wi-Fi", "the internet connection", "DNS", "the connection to the file server", "the remote desktop connection", "the firewall"],
                "de": ["das VPN", "das Büro-WLAN", "die Internetverbindung", "das DNS", "die Verbindung zum Dateiserver", "die Remotedesktop-Verbindung", "die Firewall"]},
    "hardware": {"en": ["my laptop", "the printer", "my second monitor", "the docking station", "my headset", "the laptop battery", "the keyboard"],
                 "de": ["mein Laptop", "der Drucker", "mein zweiter Monitor", "die Dockingstation", "mein Headset", "der Laptop-Akku", "die Tastatur"]},
    "feature_request": {"en": ["a CSV export option", "a dark mode", "an API for invoices", "bulk editing", "a German user interface", "scheduled emails", "custom dashboards"],
                        "de": ["eine CSV-Exportoption", "einen Dark Mode", "eine API für Rechnungen", "Mehrfachbearbeitung", "eine deutsche Benutzeroberfläche", "geplante E-Mails", "eigene Dashboards"]},
}
PATTERNS = {
    "en": ["I have a problem with {obj}", "{obj} is not working since this morning", "something is wrong with {obj}",
           "can someone check {obj}?", "{obj} keeps failing", "I need help with {obj}", "question about {obj}",
           "{obj} has been causing trouble all week", "we noticed an issue with {obj}", "is there an update on {obj}?"],
    "de": ["ich habe ein Problem mit {obj}", "{obj} funktioniert seit heute Morgen nicht", "mit {obj} stimmt etwas nicht",
           "kann jemand {obj} prüfen?", "{obj} macht ständig Probleme", "ich brauche Hilfe bei {obj}", "Frage zu {obj}",
           "{obj} macht schon die ganze Woche Ärger", "uns ist ein Problem mit {obj} aufgefallen", "gibt es Neuigkeiten zu {obj}?"],
}

URGENCY = {
    "high": {"en": ["This is urgent, production is down.", "All users are affected.", "Customers cannot pay right now!",
                    "We have a deadline today, please help asap.", "Critical: the whole team is blocked."],
             "de": ["Das ist dringend, die Produktion steht.", "Alle Nutzer sind betroffen.", "Kunden können gerade nicht bezahlen!",
                    "Wir haben heute eine Deadline, bitte schnell helfen.", "Kritisch: das ganze Team ist blockiert."]},
    "low": {"en": ["No rush, whenever you have time.", "Not urgent.", "Just a small thing for later.", "Low priority, thanks."],
            "de": ["Keine Eile, wenn Sie Zeit haben.", "Nicht dringend.", "Nur eine Kleinigkeit für später.", "Niedrige Priorität, danke."]},
    "medium": {"en": ["Could you take a look this week?", "It is a bit annoying.", "", ""],
               "de": ["Könnten Sie sich das diese Woche ansehen?", "Das ist etwas lästig.", "", ""]},
}

GREETINGS = {"en": ["Hi,", "Hello team,", "Dear support,", ""], "de": ["Hallo,", "Guten Tag,", "Liebes Support-Team,", ""]}
CLOSERS = {"en": ["Thanks!", "Best regards", "Thank you in advance.", ""], "de": ["Danke!", "Viele Grüße", "Vielen Dank im Voraus.", ""]}
DISTRACTORS = {  # sentences that mention another category, to make the task less trivial
    "en": ["I already logged in again but it did not help.", "This started after we paid the last invoice.",
           "Restarting my laptop did not change anything.", "The VPN was connected at the time.", "Maybe this is a bug?"],
    "de": ["Ich habe mich schon neu eingeloggt, hat nicht geholfen.", "Das fing nach der letzten Rechnung an.",
           "Ein Neustart des Laptops hat nichts geändert.", "Das VPN war dabei verbunden.", "Vielleicht ist das ein Bug?"],
}
FILL = {"app": ["Portal", "CRM", "Analytics", "the mobile app"], "month": ["March", "April", "May", "October"],
        "office": ["Würzburg", "Berlin", "Munich", "Hamburg"], "num": ["2", "3", "4", "4471", "8820"]}


def _typo(text: str, rng: random.Random, rate: float = 0.04) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < rate:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def make_ticket(rng: random.Random) -> dict:
    category = rng.choice(CATEGORIES)
    lang = "de" if rng.random() < 0.4 else "en"
    priority = rng.choices(PRIORITIES, weights=[0.3, 0.45, 0.25])[0]
    if rng.random() < 0.5:   # a category-specific sentence
        i = rng.randrange(len(PROBLEMS[category][lang]))
        phrasing = f"{category}/{lang}/{i}"
        problem = PROBLEMS[category][lang][i].format(**{k: rng.choice(v) for k, v in FILL.items()})
    else:                    # a generic sentence pattern around a category word
        i = rng.randrange(len(PATTERNS[lang]))
        phrasing = f"pattern/{lang}/{i}"
        problem = PATTERNS[lang][i].format(obj=rng.choice(OBJECTS[category][lang]))
    parts = [rng.choice(GREETINGS[lang]), problem[0].upper() + problem[1:] + "."]
    if rng.random() < 0.3:
        parts.append(rng.choice(DISTRACTORS[lang]))
    if rng.random() < 0.85:                      # sometimes the urgency is not written down at all
        parts.append(rng.choice(URGENCY[priority][lang]))
    parts.append(rng.choice(CLOSERS[lang]))
    text = _typo(" ".join(p for p in parts if p), rng)
    if rng.random() < 0.08:                      # inconsistent human labelling
        priority = rng.choice(PRIORITIES)
    return {"text": text, "category": category, "priority": priority, "lang": lang, "phrasing": phrasing}


def held_out_phrasings(per_group: int = 2, seed: int = 42) -> set[str]:
    """Pick `per_group` problem phrasings per category and language for the test set only, so the
    test measures how well the model handles wording it has never seen."""
    rng = random.Random(seed + 1)
    held = set()
    for cat, langs in PROBLEMS.items():
        for lang, phrasings in langs.items():
            for i in rng.sample(range(len(phrasings)), per_group):
                held.add(f"{cat}/{lang}/{i}")
    for lang, patterns in PATTERNS.items():
        for i in rng.sample(range(len(patterns)), 3):
            held.add(f"pattern/{lang}/{i}")
    return held


def make_dataset(n: int = 3000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [make_ticket(rng) for _ in range(n)]
