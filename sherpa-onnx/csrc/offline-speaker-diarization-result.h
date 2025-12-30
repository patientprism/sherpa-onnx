// sherpa-onnx/csrc/offline-speaker-diarization-result.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

class OfflineSpeakerDiarizationSegment {
 public:
  OfflineSpeakerDiarizationSegment(float start, float end, int32_t speaker,
                                   const std::string &text = {});

  // If the gap between the two segments is less than the given gap, then we
  // merge them and return a new segment. Otherwise, it returns null.
  std::optional<OfflineSpeakerDiarizationSegment> Merge(
      const OfflineSpeakerDiarizationSegment &other, float gap) const;

  float Start() const { return start_; }
  float End() const { return end_; }
  int32_t Speaker() const { return speaker_; }
  const std::string &Text() const { return text_; }
  float Duration() const { return end_ - start_; }

  void SetText(const std::string &text) { text_ = text; }

  std::string ToString() const;

 private:
  float start_;       // in seconds
  float end_;         // in seconds
  int32_t speaker_;   // ID of the speaker, starting from 0
  std::string text_;  // If not empty, it contains the speech recognition result
                      // of this segment
};

class OfflineSpeakerDiarizationResult {
 public:
  // Add a new segment
  void Add(const OfflineSpeakerDiarizationSegment &segment);

  // Number of distinct speakers contained in this object at this point
  int32_t NumSpeakers() const;

  int32_t NumSegments() const;

  // Return a list of segments sorted by segment.start time
  std::vector<OfflineSpeakerDiarizationSegment> SortByStartTime() const;

  // ans.size() == NumSpeakers().
  // ans[i] is for speaker_i and is sorted by start time
  std::vector<std::vector<OfflineSpeakerDiarizationSegment>> SortBySpeaker()
      const;

  // Set the embedding for a specific speaker
  // @param speaker_id The speaker ID (0-indexed)
  // @param embedding The embedding vector for this speaker
  void SetSpeakerEmbedding(int32_t speaker_id,
                           const std::vector<float> &embedding);

  // Get the embedding for a specific speaker
  // @param speaker_id The speaker ID (0-indexed)
  // @return The embedding vector, or an empty vector if not found
  std::vector<float> GetSpeakerEmbedding(int32_t speaker_id) const;

  // Get all speaker embeddings
  // @return A map from speaker_id to embedding vector
  const std::unordered_map<int32_t, std::vector<float>> &SpeakerEmbeddings()
      const {
    return speaker_embeddings_;
  }

  // Check if speaker embeddings are available
  bool HasSpeakerEmbeddings() const { return !speaker_embeddings_.empty(); }

  // Get the embedding dimension (0 if no embeddings are stored)
  int32_t EmbeddingDim() const;

 private:
  std::vector<OfflineSpeakerDiarizationSegment> segments_;
  std::unordered_map<int32_t, std::vector<float>> speaker_embeddings_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
