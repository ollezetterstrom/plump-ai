//! 24x suit-isomorphism augmentation.
//!
//! Plump has no trump suit, so the four suits are completely interchangeable.
//! Every training transition can therefore be relabelled by one of the 24
//! permutations of `[H, S, D, C]` without changing the game's value.

use crate::engine::{CARD_BASE, FEAT_DIM, MAX_SEQ};

#[inline]
fn permute_52_block(block: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(block.len() >= 52);
    let mut tmp = [0.0f32; 52];
    tmp.copy_from_slice(&block[..52]);
    for suit in 0..4 {
        let new_suit = perm[suit] as usize;
        for rank in 0..13 {
            block[new_suit * 13 + rank] = tmp[suit * 13 + rank];
        }
    }
}

#[inline]
fn permute_4_suits(slice: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(slice.len() >= 4);
    let tmp = [slice[0], slice[1], slice[2], slice[3]];
    for suit in 0..4 {
        slice[perm[suit] as usize] = tmp[suit];
    }
}

/// Apply `perm[old_suit] = new_suit` to a single transition in place.
///
/// Card tokens (`CARD_BASE..CARD_BASE+52`), the five 52-float card blocks,
/// the four 4-suit blocks (public voids and led suit), the three 52-float
/// opponent-hand targets, and a played action are all relabelled.
pub fn permute_transition(
    perm: &[u8; 4],
    toks: &mut [i64; MAX_SEQ],
    feats: &mut [f32; FEAT_DIM],
    opp_hands: &mut [f32; 156],
    is_play: bool,
    action: &mut i64,
) {
    for token in toks.iter_mut() {
        let t = *token;
        if (CARD_BASE as i64..CARD_BASE as i64 + 52).contains(&t) {
            let c = (t - CARD_BASE as i64) as usize;
            let new_c = perm[c / 13] as usize * 13 + (c % 13);
            *token = CARD_BASE as i64 + new_c as i64;
        }
    }

    for block_start in [0usize, 52, 104, 156, 208] {
        permute_52_block(&mut feats[block_start..block_start + 52], perm);
    }

    // 260..276: public void mask of relative players 0..3, each 4 suits.
    for player in 0..4 {
        let start = 260 + player * 4;
        permute_4_suits(&mut feats[start..start + 4], perm);
    }

    // 292..296: led suit one-hot.
    permute_4_suits(&mut feats[292..296], perm);

    for block_start in [0usize, 52, 104] {
        permute_52_block(&mut opp_hands[block_start..block_start + 52], perm);
    }

    if is_play && (0..52).contains(action) {
        let c = *action as usize;
        *action = (perm[c / 13] as i64) * 13 + (c % 13) as i64;
    }
}

/// Apply an independent random suit permutation to every row of a batch.
///
/// `toks` is `[n, MAX_SEQ]`, `feats` is `[n, FEAT_DIM]`, `opp_hands` is
/// `[n, 156]`, and `actions`/`is_play` are length `n`.
pub fn augment_batch(
    toks: &mut [i64],
    feats: &mut [f32],
    opp_hands: &mut [f32],
    actions: &mut [i64],
    is_play: &[bool],
    rng: &mut impl rand::Rng,
) {
    use rand::seq::SliceRandom;

    let n = actions.len();
    assert_eq!(is_play.len(), n);
    assert_eq!(toks.len(), n * MAX_SEQ);
    assert_eq!(feats.len(), n * FEAT_DIM);
    assert_eq!(opp_hands.len(), n * 156);

    let mut perm = [0u8, 1, 2, 3];
    for row in 0..n {
        perm.shuffle(rng);
        let t: &mut [i64; MAX_SEQ] = (&mut toks[row * MAX_SEQ..(row + 1) * MAX_SEQ])
            .try_into()
            .unwrap();
        let f: &mut [f32; FEAT_DIM] = (&mut feats[row * FEAT_DIM..(row + 1) * FEAT_DIM])
            .try_into()
            .unwrap();
        let o: &mut [f32; 156] = (&mut opp_hands[row * 156..(row + 1) * 156])
            .try_into()
            .unwrap();
        permute_transition(&perm, t, f, o, is_play[row], &mut actions[row]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BID_BASE, PLAYER_BASE};

    #[test]
    fn permutes_tokens_features_targets_and_action() {
        let mut toks = [0i64; MAX_SEQ];
        let mut feats = [0.0f32; FEAT_DIM];
        let mut opp = [0.0f32; 156];

        // Heart queen (suit 0, rank 12 => idx 11), player token, bid token.
        let hq = 11usize;
        toks[0] = 1;
        toks[1] = BID_BASE as i64 + 4;
        toks[2] = PLAYER_BASE as i64 + 2;
        toks[3] = CARD_BASE as i64 + hq as i64;
        // Heart 3 (idx 1) in own hand, spade 4 in played, void hearts for self.
        feats[1] = 1.0;
        feats[208 + 13 + 2] = 1.0;
        feats[260] = 1.0;
        feats[292] = 1.0;
        // Relative opponent 1 holds diamond 7 (suit 2 rank 7 => idx 31).
        let d7 = 31usize;
        opp[52 + d7] = 1.0;
        let mut action = hq as i64;

        // Swap hearts and spades, leave diamonds and clubs.
        let perm = [1u8, 0, 2, 3];
        permute_transition(&perm, &mut toks, &mut feats, &mut opp, true, &mut action);

        // Heart queen becomes spade queen (idx 24).
        assert_eq!(toks[3], CARD_BASE as i64 + 24);
        // Player and bid tokens untouched.
        assert_eq!(toks[2], PLAYER_BASE as i64 + 2);
        assert_eq!(toks[1], BID_BASE as i64 + 4);
        // Own hand: heart 3 -> spade 3 (idx 14).
        assert_eq!(feats[14], 1.0);
        assert_eq!(feats[1], 0.0);
        // Played: spade 4 -> heart 4 (idx 2).
        assert_eq!(feats[208 + 2], 1.0);
        assert_eq!(feats[208 + 13 + 2], 0.0);
        // Void hearts -> void spades (block 260..264, suit 1).
        assert_eq!(feats[261], 1.0);
        assert_eq!(feats[260], 0.0);
        // Led hearts -> led spades.
        assert_eq!(feats[293], 1.0);
        assert_eq!(feats[292], 0.0);
        // Diamond target unchanged.
        assert_eq!(opp[52 + d7], 1.0);
        // Action remapped.
        assert_eq!(action, 24);
    }

    #[test]
    fn inverse_permutation_restores_state() {
        let mut toks = [0i64; MAX_SEQ];
        let mut feats = [0.0f32; FEAT_DIM];
        let mut opp = [0.0f32; 156];
        for i in 0..52 {
            toks[2 + i] = CARD_BASE as i64 + i as i64;
            feats[i] = i as f32;
            opp[i] = i as f32;
        }
        let toks_orig = toks;
        let feats_orig = feats;
        let opp_orig = opp;

        let perm = [2u8, 0, 3, 1];
        let inverse = [1u8, 3, 0, 2];
        let mut action = 37i64;
        permute_transition(&perm, &mut toks, &mut feats, &mut opp, true, &mut action);
        permute_transition(&inverse, &mut toks, &mut feats, &mut opp, true, &mut action);

        assert_eq!(toks, toks_orig);
        assert_eq!(feats, feats_orig);
        assert_eq!(opp, opp_orig);
        assert_eq!(action, 37);
    }

    #[test]
    fn batch_augmentation_keeps_shapes_and_legal_values() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let n = 8;
        let mut toks = vec![0i64; n * MAX_SEQ];
        let mut feats = vec![0.0f32; n * FEAT_DIM];
        let mut opp = vec![0.0f32; n * 156];
        let mut actions = vec![0i64; n];
        let mut is_play = vec![true; n];
        for row in 0..n {
            toks[row * MAX_SEQ] = 1;
            toks[row * MAX_SEQ + 1] = BID_BASE as i64 + 3;
            toks[row * MAX_SEQ + 2] = CARD_BASE as i64 + (row as i64 * 5 % 52);
            feats[row * FEAT_DIM + row] = 1.0;
            opp[row * 156 + row] = 1.0;
            actions[row] = row as i64;
            is_play[row] = row % 2 == 0;
        }
        let mut rng = StdRng::seed_from_u64(3);
        augment_batch(
            &mut toks,
            &mut feats,
            &mut opp,
            &mut actions,
            &is_play,
            &mut rng,
        );
        for row in 0..n {
            assert_eq!(toks[row * MAX_SEQ], 1);
            assert_eq!(toks[row * MAX_SEQ + 1], BID_BASE as i64 + 3);
            assert!(actions[row] < 52);
            let sum: f32 = feats[row * FEAT_DIM..(row + 1) * FEAT_DIM].iter().sum();
            assert_eq!(sum, 1.0);
            let osum: f32 = opp[row * 156..(row + 1) * 156].iter().sum();
            assert_eq!(osum, 1.0);
        }
    }
}
