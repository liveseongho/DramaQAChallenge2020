import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics

from dataloader.dataset_multichoice import speaker_name, sos_token, eos_token, pad_token


def indices_to_words(indices, vocab):
    words = []
    for idx in indices:
        w = vocab.get_word(idx.item())
        if w in [sos_token, eos_token, pad_token]: # remove sos, eos, pad
            continue

        words.append(w)

    return words

def visual_to_words(visual, vocab):
    speaker = speaker_name[visual[0].item()]
    behavior = vocab.get_word(visual[1].item())
    emotion = vocab.get_word(visual[2].item())

    return speaker, behavior, emotion


def get_evaluator(args, model, loss_fn, metrics={}):
    # for coloring terminal output
    from termcolor import colored 

    sample_count = 0

    def _inference(evaluator, batch):
        nonlocal sample_count

        model.eval()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch, model.vocab)
            if net_inputs['subtitle'].nelement() == 0:
                import ipdb; ipdb.set_trace()  # XXX DEBUG
            y_pred = model(**net_inputs)
            batch_size = y_pred.shape[0]
            loss, stats = loss_fn(y_pred, target)

            vocab = model.vocab
            '''
            if sample_count < 100:
                print('Batch %d: data and prediction from %d to %d' % (sample_count // batch_size, sample_count, sample_count + batch_size - 1))
                
                que = net_inputs['que']
                answers = net_inputs['answers']
                # visuals = net_inputs['visual']
                script = net_inputs['filtered_sub']
                _, pred_idx = y_pred.max(dim=1)

                for i in range(batch_size):
                    targ = target[i].item()
                    pred = pred_idx[i].item()

                    ans = ['\tans %d: ' % j + ' '.join(indices_to_words(answers[i][j], vocab)) for j in range(5)]
                    if targ != pred:
                        ans[targ] = colored(ans[targ], 'green')
                        ans[pred] = colored(ans[pred], 'red')

                    print('QA', sample_count)
                    print('\tque:', *indices_to_words(que[i], vocab))
                    print('script:', *indices_to_words(script[i], vocab))
                    print(*ans, sep='\n')
                    print('\tcorrect_idx:', targ)
                    print('\tprediction:', pred)
                    # print('\tvisual:')
                    # for vis in visuals[i]:
                    #     print('\t\tspeaker: %s, behavior: %s, emotion: %s' % visual_to_words(vis, vocab))

                    sample_count += 1

                print()
            '''
            
            return loss.item(), stats, batch_size, y_pred, target  # TODO: add false_answer metric

    engine = Engine(_inference)

    metrics = {**metrics, **{
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }}
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.state

def evaluate_by_logic_level(args, model, iterator, print_total=False):
    from tqdm import tqdm

    vocab = model.vocab
    model.eval()

    cor_que_n = torch.zeros(5) # level: 1 ~ 4 (0: pad)
    all_que_n = torch.zeros(5) # level: 1 ~ 4 (0: pad)
    all_que_n[0] = 1 # Prevent DivisionByZero

    with torch.no_grad():
        for batch in tqdm(iterator, desc='Calculating accuracy by question logic level'):
            net_inputs, target = prepare_batch(args, batch, vocab)
            q_level_logic = net_inputs['q_level_logic']
            if net_inputs['subtitle'].nelement() == 0:
                import ipdb; ipdb.set_trace()  # XXX DEBUG

            y_pred = model(**net_inputs)   
            _, pred_idx = y_pred.max(dim=1)
            result = pred_idx == target

            for i, lev in enumerate(q_level_logic):
                if result[i]: # correct
                    cor_que_n[lev] += 1
                all_que_n[lev] += 1

    accuracys = cor_que_n / all_que_n.float()

    print('Accuracy by question logic level: ')
    for i in range(1, 5):
        print('Level %d: %.4f' % (i, accuracys[i]))

    if print_total:
        print('Total Accuracy:', cor_que_n.sum().item() / (all_que_n.sum().item() - 1))


def evaluate(args):
    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)
    print(args)
    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)

    metrics = get_metrics(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn, metrics)
    
    state = evaluate_once(evaluator, iterator=iters['val'])
    log_results_cmd('valid/epoch', state, 0)
    evaluate_by_logic_level(args, model, iterator=iters['val'])


def qa_similarity(args):
    from dataloader.dataset_multichoice import get_iterator
    from model.rnn import mean_along_time
    import torch.nn.functional as F

    def model(**net_inputs):
        q = net_inputs['que']
        a = net_inputs['answers']
        ql = net_inputs['que_len']
        al = net_inputs['ans_len']
        q_level_logic = net_inputs['q_level_logic']
        q = F.embedding(q, vocab)
        a = F.embedding(a, vocab)
        q = mean_along_time(q,  ql)
        a = [mean_along_time(a[:, i], al[:, i]) for i in range(5)]
        sim = torch.stack([F.cosine_similarity(q, a[i], dim=1) for i in range(5)])
        sim = sim.transpose(0, 1)

        return sim

    iters, vocab = get_iterator(args)
    iterator = iters['val']
    model.vocab = vocab
    model.eval = lambda: None
    vocab = torch.from_numpy(vocab).to(args.device)

    evaluate_by_logic_level(args, model, iterator, print_total=True)
