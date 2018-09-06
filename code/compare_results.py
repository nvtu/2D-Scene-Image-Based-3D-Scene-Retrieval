import os
import os.path as osp


def load_result(result_filepath):
    results = []
    with open(result_filepath) as f:
        r = f.readlines()
        r.pop(0)
        for inp in r:
            item = []
            img_name, temp = inp.split(',')
            item.append(img_name)
            item += [int(x) for x in temp.split()]
            results.append(item)
    return results

if __name__ == '__main__':
    results_dir = osp.join(os.getcwd(), '..', 'data', 'landmark', 'results')
    compared_dir = osp.join(os.getcwd(), '..', 'data', 'landmark', 'compared')
    r1_fp = osp.join(results_dir, 'fucking_overfit.csv')
    r2_fp = osp.join(results_dir, 'final.csv')
    r1 = load_result(r1_fp)
    r2 = load_result(r2_fp)
    num_image = len(r1)
    cfp = 'compare_results.txt'
    with open(osp.join(compared_dir, cfp), 'w') as f:
        for i in range(num_image):
            print(i)
            rl1 = r1[i][1:]
            rl2 = r2[i][1:]
            intersection = [val for val in rl1 if val in rl2]
            print(intersection)
            if len(intersection) == 0:
                f.write('{} - {}\n'.format(str(r1[i][0]), str(r2[i][0])))


