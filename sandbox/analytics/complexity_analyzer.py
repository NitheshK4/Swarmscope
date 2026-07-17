import re
from typing import List, Dict, Any
from sandbox.schemas import Message


class ComplexityAnalyzer:
    """Computes linguistic complexity metrics for multi-agent conversations.

    Provides:
    - Flesch-Kincaid Grade Level approximation (reading difficulty)
    - Type-Token Ratio (vocabulary richness)
    - Average sentence length per agent
    - Complexity degradation detection (potential loop/collapse indicator)
    - Vocabulary overlap between agents (echo chamber detection)
    """

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimates syllable count for an English word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return 1

        # Remove trailing 'e' (silent e)
        if word.endswith("e"):
            word = word[:-1]

        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        return max(syllable_count, 1)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Splits text into sentences."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _get_words(text: str) -> List[str]:
        """Extracts clean word tokens."""
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())

    def compute_flesch_kincaid(self, text: str) -> float:
        """Computes Flesch-Kincaid Grade Level for the text.
        Higher values = harder to read. Typical range: 1-18."""
        words = self._get_words(text)
        sentences = self._split_sentences(text)

        if not words or not sentences:
            return 0.0

        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        # Flesch-Kincaid Grade Level formula
        grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59
        return max(0.0, round(grade, 1))

    def compute_type_token_ratio(self, text: str) -> float:
        """Computes Type-Token Ratio (TTR) for vocabulary richness.
        Higher values = more diverse vocabulary. Range: 0.0 to 1.0."""
        words = self._get_words(text)
        if not words:
            return 0.0
        return round(len(set(words)) / len(words), 3)

    def analyze_conversation(self, messages: List[Message]) -> Dict[str, Any]:
        """Full complexity analysis across a conversation."""
        if not messages:
            return {"error": "No messages to analyze."}

        # Per-agent analysis
        agent_texts: Dict[str, List[str]] = {}
        agent_turn_complexity: Dict[str, List[Dict[str, Any]]] = {}

        for msg in messages:
            agent_texts.setdefault(msg.sender, []).append(msg.content)

            fk_grade = self.compute_flesch_kincaid(msg.content)
            ttr = self.compute_type_token_ratio(msg.content)
            words = self._get_words(msg.content)
            sentences = self._split_sentences(msg.content)
            avg_sent_len = len(words) / len(sentences) if sentences else 0.0

            agent_turn_complexity.setdefault(msg.sender, []).append({
                "turn": msg.turn,
                "fk_grade": fk_grade,
                "ttr": ttr,
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_length": round(avg_sent_len, 1)
            })

        # Agent summaries
        agent_summaries = {}
        degradation_signals = []

        for agent, turn_data in agent_turn_complexity.items():
            fk_scores = [t["fk_grade"] for t in turn_data]
            ttr_scores = [t["ttr"] for t in turn_data]

            full_text = " ".join(agent_texts[agent])
            overall_fk = self.compute_flesch_kincaid(full_text)
            overall_ttr = self.compute_type_token_ratio(full_text)

            # Detect complexity degradation (dropping reading level)
            if len(fk_scores) >= 3:
                mid = len(fk_scores) // 2
                first_half_fk = sum(fk_scores[:mid]) / mid if mid > 0 else 0
                second_half_fk = sum(fk_scores[mid:]) / (len(fk_scores) - mid)

                if first_half_fk > 0 and second_half_fk < first_half_fk * 0.7:
                    degradation_signals.append(
                        f"'{agent}' complexity dropped from FK {first_half_fk:.1f} → {second_half_fk:.1f}"
                    )

            # Detect vocabulary collapse (TTR dropping)
            if len(ttr_scores) >= 3:
                mid = len(ttr_scores) // 2
                first_half_ttr = sum(ttr_scores[:mid]) / mid if mid > 0 else 0
                second_half_ttr = sum(ttr_scores[mid:]) / (len(ttr_scores) - mid)

                if first_half_ttr > 0 and second_half_ttr < first_half_ttr * 0.6:
                    degradation_signals.append(
                        f"'{agent}' vocabulary richness collapsed from TTR {first_half_ttr:.3f} → {second_half_ttr:.3f}"
                    )

            agent_summaries[agent] = {
                "overall_fk_grade": overall_fk,
                "overall_ttr": overall_ttr,
                "avg_fk_per_turn": round(sum(fk_scores) / len(fk_scores), 1) if fk_scores else 0,
                "avg_ttr_per_turn": round(sum(ttr_scores) / len(ttr_scores), 3) if ttr_scores else 0,
                "turn_data": turn_data
            }

        # Cross-agent vocabulary overlap (echo chamber detection)
        agent_vocabularies = {}
        for agent, texts in agent_texts.items():
            combined = " ".join(texts)
            agent_vocabularies[agent] = set(self._get_words(combined))

        agents = list(agent_vocabularies.keys())
        overlap_scores = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                vocab_a, vocab_b = agent_vocabularies[a], agent_vocabularies[b]
                if vocab_a and vocab_b:
                    overlap = len(vocab_a & vocab_b) / len(vocab_a | vocab_b)
                    overlap_scores[f"{a} ↔ {b}"] = round(overlap, 3)

        return {
            "agent_summaries": agent_summaries,
            "vocabulary_overlap": overlap_scores,
            "degradation_signals": degradation_signals,
            "has_degradation": len(degradation_signals) > 0
        }
