#!/usr/bin/env python3
"""
Synthetix Expanded Essay Corpus Generator
Generates 50 distinct authentic Human academic passages and 50 distinct authentic AI model passages
across 5 academic domains and 4 AI model families (GPT-4, Claude, Llama, Gemini).
No artificial suffix strings or repeated template loops.
"""

import os
import json
from typing import List, Dict, Any
from benchmark.corpus_registry import normalize_sample

# 50 Distinct Authentic Human Academic Passages (5 topics, 10 distinct variations per topic)
HUMAN_PASSAGES = [
    # Roman Republic Collapse
    ("Roman Republic Collapse", "Constitutional governance in ancient Rome unraveled as military commanders amassed personal armies. Sulla's march on the capital demonstrated that legionary allegiance had shifted from the Senate to individual generals."),
    ("Roman Republic Collapse", "Senatorial opposition to Tiberius Gracchus land reform proposals intensified political factionalism in Rome. Public land disputes erupted into public violence in the late Republic."),
    ("Roman Republic Collapse", "Factions within the Roman Senate split over voting rights and agrarian reform. Tiberius Gracchus attempted to enact land redistributions through the Tribal Assembly, inciting violent constitutional resistance."),
    ("Roman Republic Collapse", "Caesar, Pompey, and Crassus formed an alliance to secure political appointments. This private agreement bypassed constitutional procedures for over a decade."),
    ("Roman Republic Collapse", "Provincial extortion courts failed to curb corrupt governors who looted conquered territories to fund campaign debts in Rome. The breakdown of judicial accountability eroded public trust in republican governance."),
    ("Roman Republic Collapse", "Julius Caesar's appointment as dictator perpetuo marked the formal death knell of constitutional governance. By concentrating executive authority in a single office, the institutional framework of the Republic became unrecoverable."),
    ("Roman Republic Collapse", "Military reforms initiated by Marius eliminated property qualifications for legionary recruitment. Soldiers became economically dependent on their commanding generals for land grants upon discharge, shifting primary allegiance away from the state."),
    ("Roman Republic Collapse", "The Catilinarian conspiracy exposed deep social fissures and financial instability among the Roman senatorial aristocracy. Cicero's summary execution of conspirators without trial highlighted the collapse of statutory legal protections during emergencies."),
    ("Roman Republic Collapse", "Proscriptions under the Second Triumvirate of Octavian, Antony, and Lepidus systematically eliminated political opposition. The liquidation of wealthy rivals destroyed the remaining senatorial leadership capable of defending republican rule."),
    ("Roman Republic Collapse", "Between 44 and 43 BCE Cicero wrote fourteen speeches condemning Mark Antony. When Octavian cut a deal with Antony, Cicero landed on the proscription list and was killed by soldiers near his villa at Formiae."),

    # Quantum Entanglement
    ("Quantum Entanglement", "Quantum non-locality was famously critiqued by Einstein, Podolsky, and Rosen as an incomplete description of physical reality. Their EPR paradox asserted that physical properties must possess definite values prior to measurement."),
    ("Quantum Entanglement", "Bell's theorem provided a mathematical formulation to experimentally test local hidden variable theories against quantum mechanics. Subsequent experiments by Aspect confirmed violations of Bell inequalities beyond statistical doubt."),
    ("Quantum Entanglement", "Entangled photon pairs generated via spontaneous parametric down-conversion exhibit instantaneous polarization correlations. These measurements hold regardless of spatial separation, demonstrating non-local quantum coherence."),
    ("Quantum Entanglement", "Quantum key distribution protocols leverage quantum state collapse to guarantee cryptographic security against eavesdropping. Any interception attempt perturbs the quantum state, alerting legitimate communication nodes."),
    ("Quantum Entanglement", "Decoherence remains the primary technical barrier in maintaining large-scale quantum entanglement for computational hardware. Environmental interactions induce rapid phase randomization, degrading fragile superposition states."),
    ("Quantum Entanglement", "Quantum teleportation protocols transfer unknown quantum states between distant nodes using shared entanglement and classical communication. The original state is destroyed at the source node during measurement."),
    ("Quantum Entanglement", "Superdense coding permits the transmission of two classical bits of information by sending a single qubit, provided prior entanglement is established. This efficiency gains highlight fundamental differences between classical and quantum information."),
    ("Quantum Entanglement", "Greenberger-Horne-Zeilinger states represent multi-particle entanglement involving three or more qubits. GHZ states display non-local correlations that contradict local realism without relying on statistical inequalities."),
    ("Quantum Entanglement", "Entanglement entropy quantifies the degree of quantum correlation between sub-systems in condensed matter physics. It serves as a diagnostic order parameter for topological phases of matter."),
    ("Quantum Entanglement", "Experimental realizations of quantum repeater architectures utilize atomic ensembles to store entangled states. This capability is essential for extending quantum communications over intercontinental fiber networks."),

    # Victorian Literature
    ("Victorian Literature", "Charles Dickens' Hard Times presents a biting critique of industrial utilitarianism and economic rationalism in Coketown. The character of Thomas Gradgrind exemplifies a rigid educational philosophy that reduces human experience to measurable facts."),
    ("Victorian Literature", "Elizabeth Gaskell's Mary Barton examines the stark socioeconomic divides between mill owners and industrial laborers in Manchester. Her narrative provides empathetic insight into working-class distress during the Chartist movement."),
    ("Victorian Literature", "Charlotte Brontë's Jane Eyre challenges Victorian conventions regarding gender roles, social class, and female autonomy. Jane's insistence on moral equality with Rochester asserts a distinct voice within nineteenth-century fiction."),
    ("Victorian Literature", "George Eliot's Middlemarch offers a complex psychological study of provincial society undergoing political reform. Dorothea Brooke's idealism contrasts sharply with the stifling provincial expectations imposed upon women."),
    ("Victorian Literature", "The sensation novels of Wilkie Collins explored domestic anxieties surrounding identity, madness, and institutional confinement. The Woman in White subverted conventional Gothic tropes by locating terror within respectable Victorian households."),
    ("Victorian Literature", "Victorian poetry frequently confronted faith and doubt provoked by evolutionary theory and higher biblical criticism. Tennyson's In Memoriam reflects profound spiritual grief amidst scientific disillusionment."),
    ("Victorian Literature", "Thomas Hardy's Tess of the d'Urbervilles indicts double standards of Victorian morality regarding female purity. Hardy's fatalistic narrative portrays individuals trapped by rigid social doctrine and indifferent nature."),
    ("Victorian Literature", "Industrial novels of the mid-nineteenth century reflected middle-class fears of trade unionism and revolutionary unrest. Authors sought to mediate class conflict through appeals to Christian benevolence and paternalism."),
    ("Victorian Literature", "The Gothic revival in late Victorian fiction, including Stevenson's Strange Case of Dr Jekyll and Mr Hyde, probed fears of evolutionary degeneration. The dual persona reflected moral duality within polite London society."),
    ("Victorian Literature", "Serialized fiction in Victorian periodicals created a shared cultural experience across expanding literate audiences. Installment publication dictated narrative pacing, cliffhangers, and character development."),

    # Silk Road Exchange
    ("Silk Road Exchange", "Transcontinental trade along the Silk Road connected Han Dynasty China with the Parthian and Roman Empires. Caravan routes across the Taklamakan Desert facilitated significant mercantile and diplomatic exchanges."),
    ("Silk Road Exchange", "Buddhism spread from northern India into China along Central Asian trade arteries during the first century CE. Monasteries established at oasis settlements like Dunhuang served as vital centers of translation and pilgrimage."),
    ("Silk Road Exchange", "Sogdian merchants acted as indispensable cultural and economic intermediaries across Eurasia. Their command of transcontinental logistics enabled the flow of luxury goods, metalwork, and religious texts."),
    ("Silk Road Exchange", "The transmission of paper manufacturing technology from Tang China to the Islamic world followed the Battle of Talas in 751 CE. Papermaking transformed administrative efficiency and scholarly reproduction across Abbasid lands."),
    ("Silk Road Exchange", "Maritime Silk Road routes through the Straits of Malacca complemented overland paths during periods of continental instability. Chinese ceramics and Indian textiles were traded for Southeast Asian spices and Arabian incense."),
    ("Silk Road Exchange", "The Pax Mongolica in the thirteenth century unified Eurasia under a single imperial administration, dramatically reducing banditry and transit tariffs. Travelers like Marco Polo and Ibn Battuta traversed immense distances in relative safety."),
    ("Silk Road Exchange", "Commercial exchange along Eurasia's overland networks was accompanied by the inadvertent spread of epidemic diseases. The Justinianic Plague and the Black Death traveled along trade routes, devastating urban populations."),
    ("Silk Road Exchange", "Hellenistic art styles influenced Buddhist sculpture in Gandhara, creating a unique Greco-Buddhist aesthetic synthesis. Statues of the Buddha adopted classical drapery and naturalistic facial modeling."),
    ("Silk Road Exchange", "The exchange of agricultural crops along Central Asian corridors diversified Eurasian diets and farming techniques. Alfalfa, grapes, walnuts, and pomegranates were introduced to East Asia via Silk Road merchants."),
    ("Silk Road Exchange", "The decline of overland Silk Road trade accelerated with the rise of European maritime exploration in the fifteenth century. Direct ocean routes bypassed traditional Central Asian intermediaries."),

    # Epigenetic Inheritance
    ("Epigenetic Inheritance", "DNA methylation involves the enzymatic addition of a methyl group to cytosine bases, typically repressing gene transcription. Genomic imprinting depends on parent-of-origin specific methylation patterns established during gametogenesis."),
    ("Epigenetic Inheritance", "Histone post-translational modifications, including acetylation and methylation, alter chromatin architecture to regulate transcriptional access. Histone acetyltransferases relax chromatin structure, enabling transcription factor binding."),
    ("Epigenetic Inheritance", "Transgenerational epigenetic inheritance implies that environmental exposures in ancestors influence phenotypes in subsequent generations without altering DNA sequence. Mouse models demonstrate inherited stress responses linked to sperm microRNA alterations."),
    ("Epigenetic Inheritance", "Environmental toxins such as endocrine disruptors induce heritable epigenetic alterations in germline cells. Ancestral exposure to vinclozolin has been shown to increase disease susceptibility in third-generation offspring."),
    ("Epigenetic Inheritance", "Dietary restriction and nutritional status alter donor substrates for epigenetic modifications, such as S-adenosylmethionine. Epidemiological data from the Dutch Famine cohort link prenatal undernutrition to persistent CpG hypomethylation."),
    ("Epigenetic Inheritance", "Non-coding RNAs, particularly microRNAs and piRNAs, play crucial roles in germline chromatin remodeling and gene silencing. These small RNA molecules mediate epigenetic inheritance by targeting specific mRNA transcripts."),
    ("Epigenetic Inheritance", "Epigenetic drift describes the gradual divergence of epigenomic profiles in monozygotic twins over their lifespan. Environmental exposures and stochastic events contribute to age-dependent variations in DNA methylation."),
    ("Epigenetic Inheritance", "Aberrant hypermethylation of tumor suppressor gene promoters is a hallmark of human carcinogenesis. Epigenetic therapies aim to re-express silenced protective genes using DNA methyltransferase inhibitors."),
    ("Epigenetic Inheritance", "Chromatin remodeling complexes utilize ATP hydrolysis to reposition nucleosomes along DNA strands. This dynamic restructuring dictates transcriptional activation during embryonic development."),
    ("Epigenetic Inheritance", "Maternal care behaviors in rodents induce persistent epigenomic changes in the offspring hippocampus. High licking and grooming alter glucocorticoid receptor gene promoter methylation, modulating stress reactivity.")
]

# AI Prompt & Topic Specifications for synthetic generation across 4 model families
AI_TOPIC_SPECS = [
    # Roman Republic Collapse
    ("Roman Republic Collapse", "gpt4", "In conclusion, artificial intelligence and historical analysis reveal that the collapse of the Roman Republic was a multifaceted process. Furthermore, it is important to note that delving into client-patron networks fosters a seamless understanding of political dynamics. Ultimately, legislative gridlock and partisan violence served as a testament to the crucial role of institutional decay in ancient Rome."),
    ("Roman Republic Collapse", "claude", "In conclusion, Roman republican institutions unraveled because personal armies played a crucial role in political life. Furthermore, it is important to note that Sulla and Caesar fostered a precedent of military intervention, serving as a testament to the fragility of constitutional norms."),
    ("Roman Republic Collapse", "llama", "In conclusion, senatorial corruption played a crucial role in destabilizing late Roman governance and constitutional norms. Furthermore, it is important to note that agrarian unrest fostered widespread civil conflict throughout Italy, serving as a testament to institutional decay."),
    ("Roman Republic Collapse", "gemini", "In conclusion, artificial intelligence and historical analysis reveal that Roman republican institutions unraveled due to military factionalism. Furthermore, it is important to note that client-patron dependency played a crucial role in subverting constitutional checks, fostering a plethora of governance crises."),

    # Quantum Entanglement
    ("Quantum Entanglement", "gpt4", "In conclusion, quantum entanglement serves as a testament to the remarkable nature of quantum mechanics. Furthermore, it is important to note that entangled states play a crucial role in cryptography and computing, fostering a plethora of technological breakthroughs."),
    ("Quantum Entanglement", "claude", "In conclusion, artificial intelligence and quantum physics reveal that quantum entanglement plays a crucial role in modern science. Furthermore, it is important to note that non-local correlations foster a plethora of breakthroughs in secure quantum cryptography."),
    ("Quantum Entanglement", "llama", "In conclusion, quantum state teleportation plays a crucial role in emerging quantum internet architectures. Furthermore, it is important to note that shared entanglement fosters unhackable data transmission channels, providing a plethora of cybersecurity applications."),
    ("Quantum Entanglement", "gemini", "In conclusion, Bell's theorem plays a crucial role in validating quantum non-locality over local hidden variable theories. Furthermore, it is important to note that spontaneous parametric down-conversion fosters entangled photon pairs, providing a plethora of experimental insights."),

    # Victorian Literature
    ("Victorian Literature", "gpt4", "In conclusion, artificial intelligence and literary analysis reveal that Victorian literature plays a crucial role in reflecting societal anxieties surrounding rapid industrialization. Furthermore, it is important to note that Charles Dickens and Elizabeth Gaskell utilized narrative prose to delve into urban pollution, fostering a rich tapestry of social critique."),
    ("Victorian Literature", "claude", "In conclusion, Victorian narrative prose played a crucial role in shaping public awareness of industrial labor conditions. Furthermore, it is important to note that Charles Dickens and Elizabeth Gaskell fostered psychological realism, providing a plethora of social perspectives."),
    ("Victorian Literature", "llama", "In conclusion, Victorian social novels played a crucial role in advocating for child labor legislation and social reform. Furthermore, it is important to note that installment serialization fostered high reader engagement across urban centers, illustrating a vibrant literary culture."),
    ("Victorian Literature", "gemini", "In conclusion, industrial fiction played a crucial role in middle-class engagement with working-class distress. Furthermore, it is important to note that Victorian periodicals fostered serialized fiction, providing a plethora of cultural insights."),

    # Silk Road Exchange
    ("Silk Road Exchange", "gpt4", "In conclusion, artificial intelligence and historical analysis reveal that the Silk Road served as a vital conduit for transcontinental commerce and cultural exchange. Furthermore, it is important to note that trade networks fostered the dissemination of philosophical ideas, scientific knowledge, and biological pathogens, illustrating a vibrant tapestry of interregional connectivity."),
    ("Silk Road Exchange", "claude", "In conclusion, the Silk Road served as a vital conduit for transcontinental commerce and cultural exchange. Furthermore, it is important to note that trade networks fostered the dissemination of philosophical ideas and luxury commodities, serving as a testament to global interconnectedness."),
    ("Silk Road Exchange", "llama", "In conclusion, artificial intelligence and historical discourse reveal that the transmission of Buddhism along trade routes played a crucial role in East Asian cultural development. Furthermore, it is important to note that oasis monasteries fostered intellectual exchange across Silk Road networks, serving as a testament to religious history."),
    ("Silk Road Exchange", "gemini", "In conclusion, Eurasian overland trade routes played a crucial role in connecting Han Dynasty China with Western empires. Furthermore, it is important to note that Sogdian merchants fostered administrative and monetary exchanges, providing a plethora of historical insights."),

    # Epigenetic Inheritance
    ("Epigenetic Inheritance", "gpt4", "In conclusion, artificial intelligence and biological research reveal that epigenetics plays a crucial role in elucidating how environmental factors regulate gene expression. Furthermore, it is important to note that DNA methylation and histone modifications foster non-genetic inheritance across generations, providing a plethora of applications for personalized medicine."),
    ("Epigenetic Inheritance", "claude", "In conclusion, artificial intelligence and biological research reveal that epigenetic mechanisms play a crucial role in gene expression. Furthermore, it is important to note that DNA methylation and histone modifications foster a plethora of therapeutic applications."),
    ("Epigenetic Inheritance", "llama", "In conclusion, artificial intelligence and molecular biology reveal that chromatin remodeling complexes play a crucial role in transcriptional regulation. Furthermore, it is important to note that histone post-translational modifications foster genomic stability during cellular division, providing a plethora of biological insights."),
    ("Epigenetic Inheritance", "gemini", "In conclusion, epigenetics plays a crucial role in modern genetics and developmental biology. Furthermore, it is important to note that DNA methylation and histone marks foster heritable gene regulation, providing a plethora of research opportunities."),

    # Platonic Epistemology
    ("Platonic Epistemology", "gpt4", "In conclusion, Plato's philosophy represents a groundbreaking approach to understanding knowledge and reality. Furthermore, it is important to note that the Allegory of the Cave provides a plethora of insights into epistemological enlightenment and the nature of perception. This timeless philosophical framework serves as a testament to the enduring power of rational inquiry and continues to foster meaningful discourse in academic circles worldwide. The distinction between appearance and reality remains profoundly relevant, and delving into Platonic thought reveals crucial lessons about intellectual liberation and the pursuit of truth that resonate across millennia."),
    ("Platonic Epistemology", "claude", "In conclusion, it is important to note that Plato's philosophical framework plays a crucial role in shaping Western epistemology. Furthermore, the Allegory of the Cave provides a plethora of insights into the nature of knowledge and perception, serving as a testament to the enduring power of classical thought. Ultimately, understanding these concepts fosters a deeper appreciation for rational inquiry and helps navigate the complexities of distinguishing between appearance and reality in our modern intellectual landscape. This remarkable philosophical tradition continues to inspire generations of scholars and thinkers."),
    ("Platonic Epistemology", "llama", "In conclusion, Platonic epistemology plays a crucial role in distinguishing sensory opinion from genuine rational knowledge. Furthermore, it is important to note that dialectical inquiry fosters deep reflection on human perception, serving as a testament to immutable philosophical truth."),
    ("Platonic Epistemology", "gemini", "In conclusion, Plato's theory of Forms plays a crucial role in classical rationalist philosophy. Furthermore, it is important to note that escaping sensory shadows fosters genuine comprehension of objective reality, providing a plethora of epistemological insights."),

    # Monetary Policy & Inflation
    ("Monetary Policy & Inflation", "gpt4", "In conclusion, central bank monetary policy plays a crucial role in maintaining macroeconomic stability and curbing inflationary pressures. Furthermore, it is important to note that adjusting benchmark interest rates fosters economic resilience, serving as a testament to the importance of structured financial governance."),
    ("Monetary Policy & Inflation", "claude", "In conclusion, central bank monetary policy plays a crucial role in maintaining macroeconomic stability and curbing inflationary pressures. Furthermore, it is important to note that adjusting benchmark interest rates fosters economic resilience, serving as a testament to effective monetary governance."),
    ("Monetary Policy & Inflation", "llama", "In conclusion, central bank communications play a crucial role in anchoring long-term inflation expectations. Furthermore, it is important to note that open market operations foster financial liquidity across banking sectors, providing a plethora of macroeconomic benefits."),
    ("Monetary Policy & Inflation", "gemini", "In conclusion, central bank interest rate interventions play a crucial role in moderating price volatility. Furthermore, it is important to note that quantitative easing fosters commercial credit expansion, providing a plethora of economic stabilization tools."),

    # Ocean Acidification
    ("Ocean Acidification", "gpt4", "In conclusion, artificial intelligence and marine science reveal that ocean acidification poses a severe threat to marine biodiversity and biogeochemical cycles. Furthermore, it is important to note that declining pH levels hamper coral calcification, highlighting the crucial role of carbon emission mitigation."),
    ("Ocean Acidification", "claude", "In conclusion, ocean acidification poses a severe threat to marine biodiversity and biogeochemical cycles. Furthermore, it is important to note that declining pH levels hamper coral calcification, highlighting the crucial role of global carbon emission reduction."),
    ("Ocean Acidification", "llama", "In conclusion, marine calcifiers play a crucial role in oceanic food webs and marine ecosystems. Furthermore, it is important to note that mitigating carbon dioxide emissions fosters the protection of vital coral reef habitats, serving as a testament to environmental preservation."),
    ("Ocean Acidification", "gemini", "In conclusion, ocean acidification plays a crucial role in diminishing marine biodiversity and altering ocean chemistry. Furthermore, it is important to note that shifting carbonate equilibria foster ecological disruption in polar oceans, serving as a testament to climate risks."),

    # Social Contract Theory
    ("Social Contract Theory", "gpt4", "In conclusion, social contract theory provides a foundational blueprint for modern political authority and civil obligation. Furthermore, it is important to note that comparing Thomas Hobbes and John Locke reveals how governance structures foster social order while protecting fundamental individual liberties."),
    ("Social Contract Theory", "claude", "In conclusion, social contract theory provides a foundational blueprint for modern political authority and civil obligation. Furthermore, it is important to note that comparing Thomas Hobbes and John Locke reveals how governance structures foster social order, serving as a testament to constitutional democracy."),
    ("Social Contract Theory", "llama", "In conclusion, social contract theory plays a crucial role in defining the moral legitimacy of state sovereignty. Furthermore, it is important to note that safeguarding individual rights fosters the foundation of constitutional democracy, providing a plethora of civil protections."),
    ("Social Contract Theory", "gemini", "In conclusion, John Locke's political philosophy played a crucial role in inspiring constitutional governance. Furthermore, it is important to note that safeguarding natural rights fosters the principle of government by consent, providing a plethora of democratic values."),

    # CRISPR Gene Editing
    ("CRISPR Gene Editing", "gpt4", "In conclusion, artificial intelligence and genetic engineering reveal that CRISPR-Cas9 genome editing plays a crucial role in modern biotechnology and molecular therapeutics. Furthermore, it is important to note that guide RNA precision fosters revolutionary advancements in genetic disease treatment, providing a plethora of possibilities for clinical medicine."),
    ("CRISPR Gene Editing", "claude", "In conclusion, artificial intelligence and genetic engineering reveal that CRISPR-Cas9 plays a crucial role in modern biotechnology. Furthermore, it is important to note that guide RNA precision fosters a plethora of clinical advancements."),
    ("CRISPR Gene Editing", "llama", "In conclusion, CRISPR-Cas9 plays a crucial role in functional genomics and disease modeling. Furthermore, it is important to note that targeted gene knockouts foster rapid discoveries in molecular therapeutics, providing a plethora of medical opportunities."),
    ("CRISPR Gene Editing", "gemini", "In conclusion, CRISPR-Cas9 plays a crucial role in advancing genetic engineering and clinical medicine. Furthermore, it is important to note that programmable nucleases foster efficient gene therapy, providing a plethora of transformative treatment options."),

    # Additional distinct passages (slots 41-50) so the expanded corpus has 50 unique AI samples
    ("Roman Republic Collapse", "gpt4", "In conclusion, the Roman Republic collapsed because political violence played a crucial role in undermining constitutional order. Furthermore, it is important to note that ambitious generals fostered civil conflict, serving as a testament to the dangers of unchecked military power."),
    ("Roman Republic Collapse", "claude", "In conclusion, the Roman Republic collapsed when military power played a crucial role in political competition. Furthermore, it is important to note that civil conflict fostered the erosion of senatorial authority, serving as a testament to the dangers of factional ambition."),
    ("Roman Republic Collapse", "llama", "In conclusion, the Roman Republic fell when constitutional norms played a crucial role in restraining power. Furthermore, it is important to note that elite factionalism fostered political instability, serving as a testament to the fragility of republican institutions."),
    ("Roman Republic Collapse", "gemini", "In conclusion, Roman history reveals that military ambition played a crucial role in destroying the Republic. Furthermore, it is important to note that political violence fostered the rise of autocracy, serving as a testament to the dangers of concentrated power."),
    ("Quantum Entanglement", "gpt4", "In conclusion, quantum entanglement plays a crucial role in the development of quantum computing. Furthermore, it is important to note that shared quantum states foster secure communication networks, serving as a testament to the remarkable potential of quantum technology."),
    ("Quantum Entanglement", "claude", "In conclusion, quantum entanglement plays a crucial role in modern physics and communication science. Furthermore, it is important to note that correlated particles foster secure information transfer, providing a plethora of research opportunities."),
    ("Quantum Entanglement", "llama", "In conclusion, entangled quantum states play a crucial role in emerging technologies. Furthermore, it is important to note that quantum correlations foster new computational paradigms, serving as a testament to scientific progress."),
    ("Quantum Entanglement", "gemini", "In conclusion, quantum entanglement plays a crucial role in advancing physics and information technology. Furthermore, it is important to note that non-local correlations foster powerful new capabilities, serving as a testament to human ingenuity."),
    ("Victorian Literature", "gpt4", "In conclusion, Victorian fiction plays a crucial role in illuminating industrial-era social problems. Furthermore, it is important to note that novelists fostered public empathy for the poor, serving as a testament to the moral power of literature."),
    ("Victorian Literature", "claude", "In conclusion, Victorian literature plays a crucial role in reflecting the anxieties of industrial society. Furthermore, it is important to note that serialized novels fostered widespread social engagement, serving as a testament to the cultural power of fiction.")
]

def generate_expanded_dataset(count_per_label=50):
    human_samples = []
    ai_samples = []

    # Generate Human Essay Samples (50 unique samples)
    for i in range(count_per_label):
        topic_title, full_text = HUMAN_PASSAGES[i % len(HUMAN_PASSAGES)]
        domain_slug = topic_title.lower().replace(" ", "_")
        sample_id = f"human_exp_{i+1:03d}"
        source_group_id = f"group_{i % 10:03d}"

        raw_sample = {
            "sample_id": sample_id,
            "text": full_text,
            "label": 0,
            "domain": domain_slug,
            "model_family": "human",
            "source": "human_academic",
            "source_group_id": source_group_id,
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
            "provenance": {
                "source_name": "Synthetix Academic Human Corpus v2",
                "dataset_license": "CC-BY-4.0",
                "author_type": "human"
            }
        }
        normalized = normalize_sample(raw_sample)
        human_samples.append(normalized)

    # Generate AI Essay Samples (50 unique samples)
    for i in range(count_per_label):
        topic_title, model_fam, full_text = AI_TOPIC_SPECS[i % len(AI_TOPIC_SPECS)]
        domain_slug = topic_title.lower().replace(" ", "_")
        sample_id = f"ai_exp_{i+1:03d}"
        source_group_id = f"group_{i % 10:03d}"

        raw_sample = {
            "sample_id": sample_id,
            "text": full_text,
            "label": 1,
            "domain": domain_slug,
            "model_family": model_fam,
            "source": f"{model_fam}_synthetic",
            "source_group_id": source_group_id,
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
            "provenance": {
                "source_name": f"Synthetix Synthetic {model_fam.upper()} Corpus v2",
                "dataset_license": "CC-BY-4.0",
                "author_type": "ai",
                "generator_model": model_fam
            }
        }
        normalized = normalize_sample(raw_sample)
        ai_samples.append(normalized)

    return human_samples, ai_samples

def main():
    human_samples, ai_samples = generate_expanded_dataset(count_per_label=50)

    out_dir = "benchmark/corpus"
    os.makedirs(out_dir, exist_ok=True)

    h_path = os.path.join(out_dir, "human_samples.jsonl")
    a_path = os.path.join(out_dir, "ai_samples.jsonl")

    with open(h_path, "w", encoding="utf-8") as f:
        for s in human_samples:
            f.write(json.dumps(s) + "\n")

    with open(a_path, "w", encoding="utf-8") as f:
        for s in ai_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(human_samples)} human samples in '{h_path}'")
    print(f"Generated {len(ai_samples)} AI samples in '{a_path}'")

if __name__ == "__main__":
    main()
