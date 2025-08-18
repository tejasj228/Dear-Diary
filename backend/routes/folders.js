const express = require('express');
const Folder = require('../models/Folder');
const Note = require('../models/Note');
const { checkResourceOwnership } = require('../middleware/auth');
const { invalidateUserCache } = require('../middleware/cache');

const router = express.Router();

// @desc    Get all user's folders
// @route   GET /api/folders
// @access  Private
router.get('/', async (req, res) => {
  try {
    const { includeNotesCount = false } = req.query;
    
    let folders;
    
    if (includeNotesCount === 'true') {
      folders = await Folder.aggregate([
        { $match: { userId: req.user._id } },
        {
          $lookup: {
            from: 'notes',
            let: { folderId: '$_id' },
            pipeline: [
              {
                $match: {
                  $expr: {
                    $and: [
                      { $eq: ['$folderId', '$$folderId'] },
                      { $eq: ['$isTrashed', false] }
                    ]
                  }
                }
              },
              { $count: 'count' }
            ],
            as: 'notesCount'
          }
        },
        {
          $addFields: {
            notesCount: { $ifNull: [{ $arrayElemAt: ['$notesCount.count', 0] }, 0] }
          }
        },
        { $sort: { order: 1, createdAt: 1 } }
      ]);
    } else {
      folders = await Folder.find({ userId: req.user._id })
        .sort({ order: 1, createdAt: 1 })
        .lean();
    }

    res.json({
      success: true,
      data: { folders }
    });
  } catch (error) {
    console.error('Get folders error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching folders'
    });
  }
});

// @desc    Get single folder
// @route   GET /api/folders/:id
// @access  Private
router.get('/:id', checkResourceOwnership(Folder), async (req, res) => {
  try {
    const folder = req.resource;
    
    // Get notes count
    const notesCount = await Note.countDocuments({
      folderId: folder._id,
      isTrashed: false
    });

    res.json({
      success: true,
      data: { 
        folder: {
          ...folder.toObject(),
          notesCount
        }
      }
    });
  } catch (error) {
    console.error('Get folder error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching folder'
    });
  }
});

// @desc    Create new folder
// @route   POST /api/folders
// @access  Private
router.post('/', async (req, res) => {
  try {
    const { name, color = 'purple' } = req.body;

    if (!name || !name.trim()) {
      return res.status(400).json({
        success: false,
        message: 'Folder name is required'
      });
    }

    // Check if folder name already exists for this user
    const existingFolder = await Folder.findOne({
      userId: req.user._id,
      name: name.trim()
    });

    if (existingFolder) {
      return res.status(400).json({
        success: false,
        message: 'Folder with this name already exists'
      });
    }

    // Get the highest order for new folder placement
    const maxOrderFolder = await Folder.findOne({
      userId: req.user._id
    }).sort({ order: -1 });

    const newOrder = maxOrderFolder ? maxOrderFolder.order + 1 : 0;

    const folder = await Folder.create({
      name: name.trim(),
      color,
      userId: req.user._id,
      order: newOrder
    });

    // Invalidate user's cache to ensure fresh data on next request
    invalidateUserCache(req.user._id);
    console.log('🗑️ Cache invalidated for user after folder create:', req.user._id);

    res.status(201).json({
      success: true,
      message: 'Folder created successfully',
      data: { folder }
    });
  } catch (error) {
    console.error('Create folder error:', error);
    
    // Handle folder limit error
    if (error.message.includes('Maximum folder limit')) {
      return res.status(400).json({
        success: false,
        message: error.message
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Error creating folder'
    });
  }
});

// @desc    Update folder
// @route   PUT /api/folders/:id
// @access  Private
router.put('/:id', checkResourceOwnership(Folder), async (req, res) => {
  try {
    console.log('🔧 Backend: ============ FOLDER UPDATE START ============');
    console.log('🔧 Backend: Update folder request received');
    console.log('🔧 Backend: Folder ID:', req.params.id);
    console.log('🔧 Backend: Request body:', JSON.stringify(req.body, null, 2));
    console.log('🔧 Backend: User ID:', req.user._id);
    console.log('🔧 Backend: Request headers:', req.headers);
    
    const allowedUpdates = ['name', 'color', 'order'];
    const updates = {};
    
    console.log('🔧 Backend: Processing request body...');
    Object.keys(req.body).forEach(key => {
      console.log(`🔧 Backend: Processing key "${key}" with value:`, req.body[key]);
      if (allowedUpdates.includes(key)) {
        if (key === 'name' && req.body[key]) {
          updates[key] = req.body[key].trim();
          console.log(`🔧 Backend: Added trimmed name: "${updates[key]}"`);
        } else {
          updates[key] = req.body[key];
          console.log(`🔧 Backend: Added ${key}:`, updates[key]);
        }
      } else {
        console.log(`🔧 Backend: Skipped key "${key}" (not allowed)`);
      }
    });

    console.log('🔧 Backend: Final processed updates:', JSON.stringify(updates, null, 2));

    // If updating name, check for duplicates
    if (updates.name) {
      console.log('🔧 Backend: Checking for duplicate folder names...');
      console.log('🔧 Backend: Searching for existing folder with name:', updates.name);
      console.log('🔧 Backend: Excluding folder with ID:', req.params.id);
      
      const existingFolder = await Folder.findOne({
        userId: req.user._id,
        name: updates.name,
        _id: { $ne: req.params.id }
      });

      console.log('🔧 Backend: Duplicate check result:', existingFolder ? 'FOUND DUPLICATE' : 'NO DUPLICATE');
      
      if (existingFolder) {
        console.log('🔧 Backend: Duplicate folder found:', existingFolder);
        return res.status(400).json({
          success: false,
          message: 'Folder with this name already exists'
        });
      }
    }

    console.log('🔧 Backend: Starting database update...');
    console.log('🔧 Backend: Finding folder with ID:', req.params.id);
    console.log('🔧 Backend: Applying updates:', updates);
    
    const folder = await Folder.findByIdAndUpdate(
      req.params.id,
      updates,
      { new: true, runValidators: true }
    );

    if (!folder) {
      console.log('❌ Backend: Folder not found after update');
      return res.status(404).json({
        success: false,
        message: 'Folder not found'
      });
    }

    console.log('✅ Backend: Folder updated successfully');
    console.log('✅ Backend: Updated folder data:', JSON.stringify(folder, null, 2));

    // Invalidate user's cache to ensure fresh data on next request
    console.log('🗑️ Backend: Invalidating cache for user:', req.user._id);
    invalidateUserCache(req.user._id);
    console.log('🗑️ Backend: Cache invalidated successfully');

    console.log('🔧 Backend: Sending success response...');
    res.json({
      success: true,
      message: 'Folder updated successfully',
      data: { folder }
    });
    console.log('🔧 Backend: ============ FOLDER UPDATE END ============');
  } catch (error) {
    console.error('❌ Backend: ============ FOLDER UPDATE ERROR ============');
    console.error('❌ Backend: Update folder error:', error);
    console.error('❌ Backend: Error stack:', error.stack);
    console.error('❌ Backend: Error details:', {
      message: error.message,
      name: error.name,
      code: error.code
    });
    console.error('❌ Backend: Request params:', req.params);
    console.error('❌ Backend: Request body:', req.body);
    console.error('❌ Backend: User ID:', req.user?._id);
    console.error('❌ Backend: ============ FOLDER UPDATE ERROR END ============');
    res.status(500).json({
      success: false,
      message: 'Error updating folder'
    });
  }
});

// @desc    Reorder folders
// @route   PATCH /api/folders/reorder
// @access  Private
router.patch('/reorder', async (req, res) => {
  try {
    const { folderOrders } = req.body;
    
    if (!Array.isArray(folderOrders)) {
      return res.status(400).json({
        success: false,
        message: 'folderOrders must be an array'
      });
    }

    // Update orders for each folder
    const updatePromises = folderOrders.map(({ folderId, order }) => 
      Folder.findOneAndUpdate(
        { _id: folderId, userId: req.user._id },
        { order },
        { new: true }
      )
    );

    await Promise.all(updatePromises);

    res.json({
      success: true,
      message: 'Folders reordered successfully'
    });
  } catch (error) {
    console.error('Reorder folders error:', error);
    res.status(500).json({
      success: false,
      message: 'Error reordering folders'
    });
  }
});

// @desc    Delete folder
// @route   DELETE /api/folders/:id
// @access  Private
router.delete('/:id', checkResourceOwnership(Folder), async (req, res) => {
  try {
    const folder = req.resource;
    
    // Delete folder and move notes to root
    await folder.deleteWithNotesHandling();

    // Invalidate user's cache to ensure fresh data on next request
    invalidateUserCache(req.user._id);
    console.log('🗑️ Cache invalidated for user after folder delete:', req.user._id);

    res.json({
      success: true,
      message: 'Folder deleted successfully'
    });
  } catch (error) {
    console.error('Delete folder error:', error);
    res.status(500).json({
      success: false,
      message: 'Error deleting folder'
    });
  }
});

// @desc    Get folder statistics
// @route   GET /api/folders/:id/stats
// @access  Private
router.get('/:id/stats', checkResourceOwnership(Folder), async (req, res) => {
  try {
    const folderId = req.params.id;
    
    const stats = await Note.aggregate([
      {
        $match: {
          folderId: new mongoose.Types.ObjectId(folderId),
          isTrashed: false
        }
      },
      {
        $group: {
          _id: null,
          totalNotes: { $sum: 1 },
          totalWords: { $sum: { $size: { $split: ['$content', ' '] } } },
          totalCharacters: { $sum: { $strLenCP: '$content' } },
          colorDistribution: {
            $push: '$color'
          },
          sizeDistribution: {
            $push: '$size'
          },
          lastUpdated: { $max: '$updatedAt' }
        }
      },
      {
        $project: {
          _id: 0,
          totalNotes: 1,
          totalWords: 1,
          totalCharacters: 1,
          lastUpdated: 1,
          colorDistribution: {
            $reduce: {
              input: '$colorDistribution',
              initialValue: {},
              in: {
                $mergeObjects: [
                  '$$value',
                  { $arrayToObject: [[ { k: '$$this', v: { $add: [{ $ifNull: [{ $getField: { field: '$$this', input: '$$value' } }, 0] }, 1] } } ]] }
                ]
              }
            }
          },
          sizeDistribution: {
            $reduce: {
              input: '$sizeDistribution',
              initialValue: {},
              in: {
                $mergeObjects: [
                  '$$value',
                  { $arrayToObject: [[ { k: '$$this', v: { $add: [{ $ifNull: [{ $getField: { field: '$$this', input: '$$value' } }, 0] }, 1] } } ]] }
                ]
              }
            }
          }
        }
      }
    ]);

    const folderStats = stats[0] || {
      totalNotes: 0,
      totalWords: 0,
      totalCharacters: 0,
      colorDistribution: {},
      sizeDistribution: {},
      lastUpdated: null
    };

    res.json({
      success: true,
      data: {
        folderId,
        stats: folderStats
      }
    });
  } catch (error) {
    console.error('Get folder stats error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching folder statistics'
    });
  }
});

module.exports = router;
