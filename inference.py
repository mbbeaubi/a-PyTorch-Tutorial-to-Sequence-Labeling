import torch
from utils import create_input_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViterbiDecoder():
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def decode(self, scores, lengths):
        """
        :param scores: CRF scores
        :param lengths: word sequence lengths
        :return: decoded sequences
        """
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.tagset_size), dtype=torch.long) * self.end_tag

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long)
        pointer = torch.ones((batch_size, 1),
                             dtype=torch.long) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long) * self.start_tag)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long) * self.start_tag],
                            dim=1)

        return decoded
    

def main():
    checkpoint = torch.load('BEST_checkpoint_lm_lstm_crf.pth.tar')
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    word_map = checkpoint['word_map']
    lm_vocab_size = checkpoint['lm_vocab_size']
    tag_map = checkpoint['tag_map']
    char_map = checkpoint['char_map']
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['f1']

    model.eval()

    sentence = "does this thing work"
    parsed = sentence.split()
    wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths = create_input_tensors([parsed], [[next(iter(tag_map))] * len(parsed)], word_map, char_map, tag_map)
    
    max_word_len = max(wmap_lengths.tolist())
    max_char_len = max(cmap_lengths.tolist())

    rev_tag_map = {v: k for k, v in tag_map.items()}

    # Reduce batch's padded length to maximum in-batch sequence
    # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
    wmaps = wmaps[:, :max_word_len].to(device)
    cmaps_f = cmaps_f[:, :max_char_len].to(device)
    cmaps_b = cmaps_b[:, :max_char_len].to(device)
    cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
    cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
    tmaps = tmaps[:, :max_word_len].to(device)
    wmap_lengths = wmap_lengths.to(device)
    cmap_lengths = cmap_lengths.to(device)

    # Forward prop.
    crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                cmaps_b,
                                                                                cmarkers_f,
                                                                                cmarkers_b,
                                                                                wmaps,
                                                                                tmaps,
                                                                                wmap_lengths,
                                                                                cmap_lengths)
    
    crf_scores = crf_scores.to('cpu')
    wmap_lengths_sorted = wmap_lengths_sorted.to('cpu')

    decoder = ViterbiDecoder(tag_map)
    output = decoder.decode(crf_scores, wmap_lengths_sorted)

    for idx, i in enumerate(output.data[0]):
        if (idx == len(parsed)):
            break
        
        print(parsed[idx] + " [" + rev_tag_map[i.item()] + "]")


if __name__ == '__main__':
    main()

