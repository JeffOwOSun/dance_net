import caffe
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import math

def run_solver(niter, solver, disp_interval=10, test_interval=1000, ntest_it=100):
    blobs = ('loss', 'acc')
    train_log = [] #[(it, loss, acc)]
    test_log = [] #[(it, loss, acc)]
    best_test_acc = 0.0
    best_weight_name = None
    weight_dir = tempfile.mkdtemp()

    for it in range(niter):
        #print 'iteration %d' % it
        solver.step(1)  # run a single SGD step in Caffe
        train_loss, train_acc = (solver.net.blobs[b].data.copy() for b in blobs)
        train_log.append((it, float(train_loss), float(train_acc)))

        if it % disp_interval == 0 or it + 1 == niter:
            train_disp = '%s: loss=%.3f, acc=%2d%%' % (it, train_loss,
                    np.round(100*train_acc))
            print '%3d) %s' % (it, train_disp)

        'run a thorough test'
        if it % test_interval == 0 or it + 1 == niter:
            print 'testing for %d iterations...' % ntest_it
            test_loss, test_acc = (0.0, 0.0)
            'run ntest_it iterations'
            for test_it in xrange(ntest_it):
                solver.test_nets[0].forward()
                test_loss += solver.test_nets[0].blobs['loss'].data.copy()
                test_acc += solver.test_nets[0].blobs['acc'].data.copy()

            'average them out'
            test_loss /= ntest_it
            test_acc /= ntest_it
            test_log.append((it, float(test_loss), float(test_acc)))
            test_disp = '%s: test_loss=%.3f, test_acc=%2d%%' % (it, test_loss,
                    np.round(100*test_acc))
            print '%3d) testing: %s' % (it, test_disp)

            "save the net if it's the best"
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                if best_weight_name is not None and os.path.exists(best_weight_name):
                    os.remove(best_weight_name)
                "save the new weight"
                best_weight_name = os.path.join(weight_dir,
                        'best_weight_%s.caffemodel' % it)
                solver.net.save(best_weight_name)

    return train_log, test_log, os.path.join(weight_dir, best_weight_name)

if __name__ == '__main__':
    caffe.set_device(1)
    caffe.set_mode_gpu()

    caffe_root = '/home/yushi/caffe/'
    solver_fname = caffe_root+'urop/dance/models/caffenet/solver.prototxt'
    model_fname = caffe_root+'urop/dance/models/caffenet/train_val.prototxt'
    weight_fname = caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

    print 'load the solver'
    solver = caffe.get_solver(solver_fname)

    niter = 100000
    test_interval = 1000
    ntest_it = 100
    print 'Running solvers for %d iterations...' % niter
    train_log, test_log, weight = run_solver(niter, solver,
            test_interval=test_interval, ntest_it=ntest_it)
    import shutil
    shutil.move(weight,
            caffe_root+'urop/dance/models/caffenet/'+os.path.basename(weight))

    #print train_log, test_log, weight
    import json
    with open(caffe_root+'urop/dance/models/caffenet/log.json', 'w') as f:
        f.write(json.dumps(dict(train_log=train_log, test_log=test_log,
            best_weight=weight)))

    'transpose them for plotting'
    train_iters, train_loss, train_acc = zip(*train_log)
    test_iters, test_loss, test_acc = zip(*test_log)

    fig = plt.figure(0)
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(train_iters, train_acc)
    ax2.plot(test_iters, test_acc, 'r')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('train accuracy')
    ax2.set_ylabel('test accuracy')
    #plt.show()
    plt.savefig(caffe_root+'urop/dance/models/caffenet/train_val.png')




